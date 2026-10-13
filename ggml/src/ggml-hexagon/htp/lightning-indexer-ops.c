#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "hvx-copy.h"
#include "htp-vtcm.h"
#include "work-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-quant.h"
#include "hvx-fa-kernels.h"

#include "lightning-indexer-ops.h"

struct htp_lid_context {
    const struct htp_ops_context * octx;

    uint32_t n_blocks;
    uint32_t blocks_per_thread;

    // current stream and token tile
    uint32_t is;
    uint32_t im3;
    uint32_t t0;
    uint32_t nt;

    struct htp_lid_vtcm_layout L;

    uint8_t * spad_q;
    uint8_t * spad_w;
    uint8_t * spad_qs;
    uint8_t * spad_k;
    uint8_t * spad_kt;
    uint8_t * spad_m;
    uint8_t * spad_o;
};

// Splat every Q element (f16) and every head weight (f32) into a full vector.
// Work is split over (token, head) pairs.
static void lid_prep_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lid_context *       lctx = (struct htp_lid_context *) data;
    const struct htp_ops_context * octx = lctx->octx;
    const struct htp_tensor *      q    = octx->src[0];
    const struct htp_tensor *      w    = octx->src[2];

    const struct htp_lid_vtcm_layout * L = &lctx->L;

    const uint32_t D      = q->ne[0];
    const uint32_t n_head = q->ne[1];

    const uint32_t n_items = lctx->nt * n_head;
    const uint32_t chunk   = (n_items + nth - 1) / nth;
    const uint32_t i0      = ith * chunk;
    const uint32_t i1      = MIN(i0 + chunk, n_items);

    if (i0 >= i1) {
        return;
    }

    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];
    dma_queue *               dma_q = octx->ctx->dma[ith];

    const uint32_t w_row_size = n_head * sizeof(float);

    uint8_t * qs = lctx->spad_qs + ith * L->size_qs;
    uint8_t * ws = qs + n_head * L->size_qs_row;
    uint8_t * qh = ws + hex_round_up(w_row_size, 128);

    uint32_t t  = i0 / n_head;
    uint32_t ha = i0 - t * n_head;

    for (uint32_t i = i0; i < i1; ++t, ha = 0) {
        const uint32_t hb = MIN(n_head, ha + (i1 - i));
        const uint32_t it = lctx->t0 + t;

        dma_queue_push(dma_q, dma_make_data(qs, q->data + it * q->nb[2] + lctx->is * q->nb[3] + ha * q->nb[1]),
                       L->size_qs_row, q->nb[1], D * sizeof(float), hb - ha);
        dma_queue_push(dma_q, dma_make_data(ws, w->data + it * w->nb[1] + lctx->is * w->nb[3]),
                       w_row_size, w_row_size, w_row_size, 1);
        dma_queue_pop(dma_q);
        dma_queue_pop(dma_q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) it);

        HVX_Vector * restrict q_sp = (HVX_Vector *) lctx->spad_q + (t * n_head + ha) * D;
        for (uint32_t h = ha; h < hb; ++h) {
            hvx_copy_f16_f32_aa(qh, qs + (h - ha) * L->size_qs_row, D);

            const HVX_Vector * restrict qv = (const HVX_Vector *) qh;
            for (uint32_t d0 = 0; d0 < D; d0 += VLEN_FP16) {
                const HVX_Vector v  = qv[d0 / VLEN_FP16];
                const uint32_t   nd = MIN(VLEN_FP16, D - d0);
                for (uint32_t j = 0; j < nd; ++j) {
                    *q_sp++ = hvx_vec_repl_f16(Q6_V_vror_VR(v, j * SIZEOF_FP16));
                }
            }
        }

        const HVX_Vector * restrict wv    = (const HVX_Vector *) ws;
        HVX_Vector *       restrict w_dst = (HVX_Vector *) (lctx->spad_w + t * n_head * VLEN);
        for (uint32_t h = ha; h < hb; ++h) {
            w_dst[h] = hvx_vec_repl4(Q6_V_vror_VR(wv[h / VLEN_FP32], (h % VLEN_FP32) * SIZEOF_FP32));
        }

        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) it);

        i += hb - ha;
    }
}

// Q8_0 row to f16 in place, 4 blocks (128 values) per step.
// Go backwards: the f16 output of a group is wider than its input and covers the input of the next group.
static inline void lid_dequant_row_q8_0(uint8_t * restrict row, uint32_t D) {
    const int32_t nb = D / QK8_0;

    const HVX_VectorPred q32 = Q6_Q_vsetq_R(32);
    const HVX_VectorPred q64 = Q6_Q_vsetq_R(64);
    const HVX_VectorPred q96 = Q6_Q_vsetq_R(96);

    for (int32_t g = (nb - 1) / 4; g >= 0; --g) {
        const uint8_t *  src = row + g * 4 * sizeof(block_q8_0);
        const HVX_Vector v0  = *(const HVX_UVector *) src;
        const HVX_Vector v1  = *(const HVX_UVector *) (src + VLEN);

        // quants of block i start at byte 34 * i + 2, move them to byte 32 * i
        HVX_Vector qs = Q6_V_vmux_QVV(q96, Q6_V_valign_VVR(v1, v0, 6), Q6_V_valign_VVR(v1, v0, 8));
        qs            = Q6_V_vmux_QVV(q64, Q6_V_valign_VVR(v1, v0, 4), qs);
        qs            = Q6_V_vmux_QVV(q32, Q6_V_valign_VVR(v1, v0, 2), qs);

        const HVX_Vector s01 = Q6_V_vmux_QVV(q64, hvx_vec_repl_f16(v0), hvx_vec_repl_f16(Q6_V_valign_VVR(v1, v0, 34)));
        const HVX_Vector s23 = Q6_V_vmux_QVV(q64, hvx_vec_repl_f16(Q6_V_valign_VVR(v1, v0, 68)), hvx_vec_repl_f16(Q6_V_valign_VVR(v1, v0, 102)));

        const HVX_VectorPair qh = Q6_Wh_vunpack_Vb(qs);

        HVX_Vector * dst = (HVX_Vector *) (row + g * 2 * VLEN);
        dst[0] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(qh)), s01));
        if (g * 4 + 2 < nb) {
            dst[1] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(qh)), s23));
        }
    }
}

static inline void lid_shuff_pass(HVX_Vector * restrict dst, const HVX_Vector * restrict src) {
    for (uint32_t j = 0; j < 32; ++j) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(src[j + 32], src[j], -SIZEOF_FP16);
        dst[2 * j + 0]   = Q6_V_lo_W(p);
        dst[2 * j + 1]   = Q6_V_hi_W(p);
    }
}

// 64x64 f16 transpose: each shuffle pass rotates the 12-bit (row, lane) index by one bit, six passes swap row and lane
static inline void lid_transpose_64x64_f16(HVX_Vector * restrict dst,
                                           const uint8_t * restrict src,
                                           size_t src_stride,
                                           HVX_Vector * restrict t0,
                                           HVX_Vector * restrict t1) {
    for (uint32_t j = 0; j < 32; ++j) {
        const HVX_Vector r0 = *(const HVX_Vector *) (src + j * src_stride);
        const HVX_Vector r1 = *(const HVX_Vector *) (src + (j + 32) * src_stride);
        HVX_VectorPair   p  = Q6_W_vshuff_VVR(r1, r0, -SIZEOF_FP16);
        t0[2 * j + 0]       = Q6_V_lo_W(p);
        t0[2 * j + 1]       = Q6_V_hi_W(p);
    }
    lid_shuff_pass(t1, t0);
    lid_shuff_pass(t0, t1);
    lid_shuff_pass(t1, t0);
    lid_shuff_pass(t0, t1);
    lid_shuff_pass(dst, t0);
}

#if __HVX_ARCH__ < 79
// qf32 accumulators, converted to sf once per head
static inline HVX_VectorPair lid_mpyacc(HVX_VectorPair acc, HVX_Vector x, HVX_Vector y) {
    HVX_VectorPair m = Q6_Wqf32_vmpy_VhfVhf(x, y);
    return Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(acc), Q6_V_hi_W(m)),
                            Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(acc), Q6_V_lo_W(m)));
}
static inline HVX_Vector lid_acc_to_sf(HVX_Vector v) {
    return Q6_Vsf_equals_Vqf32(v);
}
#else
static inline HVX_VectorPair lid_mpyacc(HVX_VectorPair acc, HVX_Vector x, HVX_Vector y) {
    return Q6_Wsf_vmpyacc_WsfVhfVhf(acc, x, y);
}
static inline HVX_Vector lid_acc_to_sf(HVX_Vector v) {
    return v;
}
#endif

// out += relu(s) * w, for the even and odd kv lanes
static inline HVX_VectorPair lid_head_acc(HVX_VectorPair out, HVX_VectorPair s, HVX_Vector w) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector e = HVX_OP_ADD_F32(Q6_V_lo_W(out), HVX_OP_MUL_F32(Q6_Vsf_vmax_VsfVsf(lid_acc_to_sf(Q6_V_lo_W(s)), zero), w));
    HVX_Vector o = HVX_OP_ADD_F32(Q6_V_hi_W(out), HVX_OP_MUL_F32(Q6_Vsf_vmax_VsfVsf(lid_acc_to_sf(Q6_V_hi_W(s)), zero), w));
    return Q6_W_vcombine_VV(o, e);
}

// One output row (token) of a kv block: sum_h relu(q_h . k) * w_h + mask
// kt holds the K block transposed (one f16 vector of 64 kv per d), q_t holds the Q splats of the token.
static inline void lid_compute_row(float * restrict o,
                                   const HVX_Vector * restrict q_t,
                                   const HVX_Vector * restrict w_t,
                                   const HVX_Vector * restrict kt,
                                   const __fp16 * restrict m,
                                   uint32_t D,
                                   uint32_t n_head) {
    const HVX_Vector zero = Q6_V_vzero();

    // f16 x f16 products land in (even kv, odd kv) lane order
    HVX_VectorPair out = Q6_W_vzero();

    uint32_t h = 0;
    for (; h + 4 <= n_head; h += 4) {
        const HVX_Vector * restrict q0 = q_t + h * D;
        const HVX_Vector * restrict q1 = q0 + D;
        const HVX_Vector * restrict q2 = q1 + D;
        const HVX_Vector * restrict q3 = q2 + D;

        HVX_VectorPair s0 = Q6_W_vzero();
        HVX_VectorPair s1 = Q6_W_vzero();
        HVX_VectorPair s2 = Q6_W_vzero();
        HVX_VectorPair s3 = Q6_W_vzero();

        for (uint32_t d = 0; d < D; ++d) {
            const HVX_Vector k = kt[d];
            s0 = lid_mpyacc(s0, k, q0[d]);
            s1 = lid_mpyacc(s1, k, q1[d]);
            s2 = lid_mpyacc(s2, k, q2[d]);
            s3 = lid_mpyacc(s3, k, q3[d]);
        }

        out = lid_head_acc(out, s0, w_t[h + 0]);
        out = lid_head_acc(out, s1, w_t[h + 1]);
        out = lid_head_acc(out, s2, w_t[h + 2]);
        out = lid_head_acc(out, s3, w_t[h + 3]);
    }
    for (; h < n_head; ++h) {
        const HVX_Vector * restrict q0 = q_t + h * D;

        HVX_VectorPair s0 = Q6_W_vzero();
        for (uint32_t d = 0; d < D; ++d) {
            s0 = lid_mpyacc(s0, kt[d], q0[d]);
        }
        out = lid_head_acc(out, s0, w_t[h]);
    }

    const HVX_VectorPair out_nat = Q6_W_vshuff_VVR(Q6_V_hi_W(out), Q6_V_lo_W(out), -SIZEOF_FP32);

    HVX_Vector acc0 = Q6_V_lo_W(out_nat);
    HVX_Vector acc1 = Q6_V_hi_W(out_nat);

    // qf32 math does not keep inf: add a finite mask, then put -inf back
    const HVX_Vector     m_f16 = *(const HVX_Vector *) m;
    const HVX_VectorPred m_inf = Q6_Q_vcmp_eq_VhVh(m_f16, Q6_Vh_vsplat_R(0xFC00));

    const HVX_VectorPair m_f32 = hvx_vec_f16_to_f32(Q6_V_vmux_QVV(m_inf, zero, m_f16));
    const HVX_VectorPair f_f32 = hvx_vec_f16_to_f32(Q6_V_vmux_QVV(m_inf, Q6_Vh_vsplat_R(0x3C00), zero));

    const HVX_Vector ninf = Q6_V_vsplat_R(0xFF800000);

    acc0 = HVX_OP_ADD_F32(acc0, Q6_V_lo_W(m_f32));
    acc1 = HVX_OP_ADD_F32(acc1, Q6_V_hi_W(m_f32));
    acc0 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_lo_W(f_f32), zero), ninf, acc0);
    acc1 = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_hi_W(f_f32), zero), ninf, acc1);

    HVX_Vector * restrict vo = (HVX_Vector *) o;
    vo[0] = acc0;
    vo[1] = acc1;
}

// Each thread owns a range of kv blocks and streams K / mask in, output out (double buffered)
static void lid_compute_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lid_context *       lctx = (struct htp_lid_context *) data;
    const struct htp_ops_context * octx = lctx->octx;
    const struct htp_tensor *      q    = octx->src[0];
    const struct htp_tensor *      k    = octx->src[1];
    const struct htp_tensor *      m    = octx->src[3];
    const struct htp_tensor *      dst  = octx->dst;

    const uint32_t b0 = lctx->blocks_per_thread * ith;
    const uint32_t b1 = MIN(b0 + lctx->blocks_per_thread, lctx->n_blocks);

    if (b0 >= b1) {
        return;
    }

    const struct htp_lid_vtcm_layout * L = &lctx->L;

    const uint32_t D      = q->ne[0];
    const uint32_t n_head = q->ne[1];
    const uint32_t n_kv   = k->ne[2];
    const uint32_t nt     = lctx->nt;

    const bool     is_q8_0    = (k->type == HTP_TYPE_Q8_0);
    const uint32_t k_row_size = htp_tensor_get_row_size(k->type, D);

    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbm1 = m->nb[1];
    const uint32_t nb1  = dst->nb[1];

    const dma_addr_t k_base = k->data   + lctx->is  * k->nb[3];
    const dma_addr_t m_base = m->data   + lctx->t0  * nbm1 + lctx->im3 * m->nb[3];
    const dma_addr_t d_base = dst->data + lctx->t0  * nb1  + lctx->is  * dst->nb[3];

    uint8_t * spad_k = lctx->spad_k + ith * 2 * L->size_k_buf;
    uint8_t * spad_m = lctx->spad_m + ith * 2 * L->size_m_buf;
    uint8_t * spad_o = lctx->spad_o + ith * 2 * L->size_o_buf;

    HVX_Vector * kt    = (HVX_Vector *) (lctx->spad_kt + ith * L->size_kt);
    HVX_Vector * kt_t0 = kt + hex_round_up(D, HTP_LID_KV_BLOCK);
    HVX_Vector * kt_t1 = kt_t0 + HTP_LID_KV_BLOCK;

    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];
    dma_queue *               dma_q = octx->ctx->dma[ith];

    for (uint32_t b = b0, i = 0; b < b1 && i < 2; ++b, ++i) {
        const uint32_t kv0 = b * HTP_LID_KV_BLOCK;
        const uint32_t nkv = MIN(HTP_LID_KV_BLOCK, n_kv - kv0);

        // dummy output writeback to keep the queue order
        dma_queue_push(dma_q, dma_make_data(d_base, spad_o + i * L->size_o_buf),
                       nb1, L->size_o_row, nkv * sizeof(float), 0);
        dma_queue_push(dma_q, dma_make_data(spad_k + i * L->size_k_buf, k_base + kv0 * nbk2),
                       L->size_k_row, nbk2, k_row_size, nkv);
        dma_queue_push(dma_q, dma_make_data(spad_m + i * L->size_m_buf, m_base + kv0 * sizeof(__fp16)),
                       L->size_m_row, nbm1, nkv * sizeof(__fp16), nt);
    }

    for (uint32_t b = b0; b < b1; ++b) {
        const uint32_t kv0 = b * HTP_LID_KV_BLOCK;
        const uint32_t nkv = MIN(HTP_LID_KV_BLOCK, n_kv - kv0);

        uint8_t * o_buf = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;
        uint8_t * k_buf = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;
        uint8_t * m_buf = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;

        if (is_q8_0) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) b);
            for (uint32_t r = 0; r < nkv; ++r) {
                lid_dequant_row_q8_0(k_buf + r * L->size_k_row, D);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) b);
        }

        // rows past nkv hold stale data, they only reach kv lanes that are not written back
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) b);
        for (uint32_t d0 = 0; d0 < D; d0 += HTP_LID_KV_BLOCK) {
            lid_transpose_64x64_f16(kt + d0, k_buf + d0 * sizeof(__fp16), L->size_k_row, kt_t0, kt_t1);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) b);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);
        for (uint32_t t = 0; t < nt; ++t) {
            lid_compute_row((float *) (o_buf + t * L->size_o_row),
                            (const HVX_Vector *) lctx->spad_q + t * n_head * D,
                            (const HVX_Vector *) (lctx->spad_w + t * n_head * VLEN),
                            kt,
                            (const __fp16 *) (m_buf + t * L->size_m_row),
                            D, n_head);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);

        dma_queue_push(dma_q, dma_make_data(d_base + kv0 * sizeof(float), o_buf),
                       nb1, L->size_o_row, nkv * sizeof(float), nt);

        if (b + 2 < b1) {
            const uint32_t kv2  = kv0 + 2 * HTP_LID_KV_BLOCK;
            const uint32_t nkv2 = MIN(HTP_LID_KV_BLOCK, n_kv - kv2);

            dma_queue_push(dma_q, dma_make_data(k_buf, k_base + kv2 * nbk2),
                           L->size_k_row, nbk2, k_row_size, nkv2);
            dma_queue_push(dma_q, dma_make_data(m_buf, m_base + kv2 * sizeof(__fp16)),
                           L->size_m_row, nbm1, nkv2 * sizeof(__fp16), nt);
        }
    }

    dma_queue_flush(dma_q);
}

int op_lightning_indexer(struct htp_ops_context * octx) {
    const struct htp_tensor * q = octx->src[0];
    const struct htp_tensor * k = octx->src[1];
    const struct htp_tensor * m = octx->src[3];

    if (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // no multi-device split yet, device 0 does all the work
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx != 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_lid_kernel_params * kparams = (const struct htp_lid_kernel_params *) octx->kernel_params;

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_lid_context lctx;
    lctx.octx              = octx;
    lctx.n_blocks          = kparams->n_kv_blocks;
    lctx.blocks_per_thread = kparams->blocks_per_thread;

    htp_lid_vtcm_layout_build(&lctx.L, q->ne[0], q->ne[1], kparams->n_tokens_tile, kparams->n_threads);

    uint8_t * const base = octx->ctx->vtcm_base;

    lctx.spad_q  = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_q);
    lctx.spad_w  = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_w);
    lctx.spad_qs = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_qs);
    lctx.spad_k  = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_k);
    lctx.spad_kt = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_kt);
    lctx.spad_m  = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_m);
    lctx.spad_o  = VTCM_LAYOUT_PTR(uint8_t, base, lctx.L.off_o);

    const uint32_t n_tokens  = q->ne[2];
    const uint32_t n_stream  = q->ne[3];
    const uint32_t n_tile    = kparams->n_tokens_tile;
    const uint32_t n_threads = kparams->n_threads;

    for (uint32_t is = 0; is < n_stream; ++is) {
        lctx.is  = is;
        lctx.im3 = is % m->ne[3];

        for (uint32_t t0 = 0; t0 < n_tokens; t0 += n_tile) {
            lctx.t0 = t0;
            lctx.nt = MIN(n_tile, n_tokens - t0);

            work_queue_run(octx->ctx->work_queue, lid_prep_thread, &lctx, MIN(n_threads, lctx.nt * q->ne[1]));
            work_queue_run(octx->ctx->work_queue, lid_compute_thread, &lctx, n_threads);
        }
    }

    return HTP_STATUS_OK;
}
