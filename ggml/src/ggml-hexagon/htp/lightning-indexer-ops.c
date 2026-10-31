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
#include "hmx-utils.h"
#include "hmx-fa-kernels.h"
#include "hmx-queue.h"

#include "lightning-indexer-ops.h"

struct htp_lid_context {
    const struct htp_ops_context *       octx;
    const struct htp_lid_kernel_params * kp;

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

    // nth is min(n_threads, n_items): the same chunk as a split over n_threads
    const uint32_t n_items = lctx->nt * n_head;
    const uint32_t chunk   = fastdiv(n_items + octx->n_threads - 1, &octx->n_threads_div);
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

    uint32_t t  = fastdiv(i0, &lctx->kp->div_n_head);
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

#if __HVX_ARCH__ < 79
#define LID_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#else
#define LID_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#endif

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

// Q4_0 blocks i, i + 1 (v loaded at block i) to signed quants: block i in bytes 0..31, block i + 1 in bytes 32..63.
// qs[j] holds value j (low nibble) and j + 16 (high nibble). lo16 selects bytes 0..15 and 32..47.
static inline HVX_Vector lid_q4_0_pair_quants(HVX_Vector v, HVX_VectorPred q32, HVX_VectorPred lo16) {
    const HVX_Vector qs = Q6_V_vmux_QVV(q32, Q6_V_vror_VR(v, sizeof(ggml_half)),
                                        Q6_V_vror_VR(v, VLEN - 32 + sizeof(block_q4_0) + sizeof(ggml_half)));
    const HVX_Vector lo = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(qs, Q6_Vb_vsplat_R(0x0F)), Q6_Vb_vsplat_R(8));
    const HVX_Vector hi = Q6_Vb_vsub_VbVb(Q6_Vub_vlsr_VubR(qs, 4), Q6_Vb_vsplat_R(8));
    return Q6_V_vmux_QVV(lo16, lo, Q6_V_vlalign_VVR(hi, Q6_V_vzero(), 16));
}

// Q4_0 row to f16 in place, 2 blocks (64 values) per step. Backwards, as for Q8_0.
static inline void lid_dequant_row_q4_0(uint8_t * restrict row, uint32_t D) {
    const int32_t nb = D / QK4_0;

    const HVX_VectorPred q32  = Q6_Q_vsetq_R(32);
    const HVX_VectorPred q64  = Q6_Q_vsetq_R(64);
    const HVX_VectorPred lo16 = Q6_Q_or_QQ(Q6_Q_vsetq_R(16), Q6_Q_and_QQn(Q6_Q_vsetq_R(48), q32));

    for (int32_t p = (nb - 1) / 2; p >= 0; --p) {
        const HVX_Vector v  = *(const HVX_UVector *) (row + p * 2 * sizeof(block_q4_0));
        const HVX_Vector q  = lid_q4_0_pair_quants(v, q32, lo16);
        const HVX_Vector d1 = 2 * p + 1 < nb ? hvx_vec_repl_f16(Q6_V_vror_VR(v, sizeof(block_q4_0))) : Q6_V_vzero();
        const HVX_Vector d  = Q6_V_vmux_QVV(q64, hvx_vec_repl_f16(v), d1);

        ((HVX_Vector *) row)[p] = LID_MUL_F16(Q6_Vhf_equals_Vh(Q6_V_lo_W(Q6_Wh_vunpack_Vb(q))), d);
    }
}

// Quantized kv block to f16 rows in place. Not inlined: the dequant constants would stay live across the compute loop.
static __attribute__((noinline)) void lid_dequant_block(uint8_t * restrict k_buf, size_t row_stride, uint32_t nkv, uint32_t D, bool q4) {
    if (q4) {
        for (uint32_t r = 0; r < nkv; ++r) {
            lid_dequant_row_q4_0(k_buf + r * row_stride, D);
        }
    } else {
        for (uint32_t r = 0; r < nkv; ++r) {
            lid_dequant_row_q8_0(k_buf + r * row_stride, D);
        }
    }
}

static inline void lid_convert_row_f32_to_f16(uint8_t * row, uint32_t D) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector * src = (const HVX_Vector *) row;
    HVX_Vector * dst = (HVX_Vector *) row;

    for (uint32_t d = 0; d < D; d += 64) {
        dst[d / 64] = hvx_vec_f32_to_f16(src[d / 32], d + 32 < D ? src[d / 32 + 1] : zero);
    }
}

// Q8_0 gather transpose limits: tmp (two 64x64 scratch tiles) holds D / 2 pairs, gather offsets are 16 bit
static inline bool lid_q8_0_gather_ok(size_t row_size, uint32_t D) {
    return D <= 4 * HTP_LID_KV_BLOCK && (HTP_LID_KV_BLOCK - 1) * row_size + (D / QK8_0) * sizeof(block_q8_0) <= 32767;
}

// Transpose a raw Q8_0 kv block: one halfword gather reads q[d], q[d + 1] of all 64 rows, then split, widen to f16 and scale.
// Lanes of missing rows hold stale data; they are not stored.
static inline void lid_transpose_q8_0_gather(HVX_Vector * restrict kt,
                                             HVX_Vector * restrict tmp,
                                             HVX_Vector * restrict sc,
                                             const uint8_t * restrict k_buf,
                                             size_t row_size,
                                             HVX_Vector offsets,
                                             uint32_t D) {
    const uint32_t mu = HTP_LID_KV_BLOCK * row_size;
    for (uint32_t b = 0; b < D / QK8_0; ++b) {
        Q6_vgather_ARMVh(sc + b, (size_t) (k_buf + b * sizeof(block_q8_0)), mu, offsets);
    }
    for (uint32_t d = 0; d < D; d += 2) {
        const uint32_t off = (d / QK8_0) * sizeof(block_q8_0) + sizeof(ggml_half) + d % QK8_0;
        Q6_vgather_ARMVh(tmp + d / 2, (size_t) (k_buf + off), mu, offsets);
    }
    hvx_gather_sync(sc);
    hvx_gather_sync(tmp);

    for (uint32_t d = 0; d < D; d += 2) {
        const HVX_Vector     s  = sc[d / QK8_0];
        const HVX_VectorPair qh = Q6_Wh_vunpack_Vb(tmp[d / 2]);
        const HVX_VectorPair qd = Q6_W_vdeal_VVR(Q6_V_hi_W(qh), Q6_V_lo_W(qh), -2);
        kt[d + 0] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_lo_W(qd)), s));
        kt[d + 1] = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_V_hi_W(qd)), s));
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
                                           uint32_t nkv,
                                           HVX_Vector * restrict t0,
                                           HVX_Vector * restrict t1) {
    if (nkv >= HTP_LID_KV_BLOCK) {
        for (uint32_t j = 0; j < 32; ++j) {
            const HVX_Vector r0 = *(const HVX_Vector *) (src + j * src_stride);
            const HVX_Vector r1 = *(const HVX_Vector *) (src + (j + 32) * src_stride);
            HVX_VectorPair   p  = Q6_W_vshuff_VVR(r1, r0, -SIZEOF_FP16);
            t0[2 * j + 0]       = Q6_V_lo_W(p);
            t0[2 * j + 1]       = Q6_V_hi_W(p);
        }
    } else {
        const HVX_Vector zero = Q6_V_vzero();
        for (uint32_t j = 0; j < 32; ++j) {
            const HVX_Vector r0 = (j < nkv) ? *(const HVX_Vector *) (src + j * src_stride) : zero;
            const HVX_Vector r1 = (j + 32 < nkv) ? *(const HVX_Vector *) (src + (j + 32) * src_stride) : zero;
            HVX_VectorPair   p  = Q6_W_vshuff_VVR(r1, r0, -SIZEOF_FP16);
            t0[2 * j + 0]       = Q6_V_lo_W(p);
            t0[2 * j + 1]       = Q6_V_hi_W(p);
        }
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

// Store one output row: (even, odd) lanes back to natural order, add the mask
static inline void lid_store_row(float * restrict o, HVX_VectorPair out, const __fp16 * restrict m) {
    const HVX_Vector zero = Q6_V_vzero();

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

// One output row (token) of a kv block: sum_h relu(q_h . k) * w_h + mask
// kt holds the K block transposed (one f16 vector of 64 kv per d), q_t holds the Q splats of the token.
static inline void lid_compute_row(float * restrict o,
                                   const HVX_Vector * restrict q_t,
                                   const HVX_Vector * restrict w_t,
                                   const HVX_Vector * restrict kt,
                                   const __fp16 * restrict m,
                                   uint32_t D,
                                   uint32_t n_head) {
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

    lid_store_row(o, out, m);
}

// Same as lid_compute_row for two kv blocks at once: each Q splat load feeds both blocks
static inline void lid_compute_row2(float * restrict o0,
                                    float * restrict o1,
                                    const HVX_Vector * restrict q_t,
                                    const HVX_Vector * restrict w_t,
                                    const HVX_Vector * restrict kt0,
                                    const HVX_Vector * restrict kt1,
                                    const __fp16 * restrict m0,
                                    const __fp16 * restrict m1,
                                    uint32_t D,
                                    uint32_t n_head) {
    HVX_VectorPair out0 = Q6_W_vzero();
    HVX_VectorPair out1 = Q6_W_vzero();

    uint32_t h = 0;
    for (; h + 4 <= n_head; h += 4) {
        const HVX_Vector * restrict q0 = q_t + h * D;
        const HVX_Vector * restrict q1 = q0 + D;
        const HVX_Vector * restrict q2 = q1 + D;
        const HVX_Vector * restrict q3 = q2 + D;

        HVX_VectorPair a0 = Q6_W_vzero();
        HVX_VectorPair a1 = Q6_W_vzero();
        HVX_VectorPair a2 = Q6_W_vzero();
        HVX_VectorPair a3 = Q6_W_vzero();
        HVX_VectorPair b0 = Q6_W_vzero();
        HVX_VectorPair b1 = Q6_W_vzero();
        HVX_VectorPair b2 = Q6_W_vzero();
        HVX_VectorPair b3 = Q6_W_vzero();

        for (uint32_t d = 0; d < D; ++d) {
            const HVX_Vector k0 = kt0[d];
            const HVX_Vector k1 = kt1[d];
            const HVX_Vector v0 = q0[d];
            const HVX_Vector v1 = q1[d];
            const HVX_Vector v2 = q2[d];
            const HVX_Vector v3 = q3[d];
            a0 = lid_mpyacc(a0, k0, v0);
            b0 = lid_mpyacc(b0, k1, v0);
            a1 = lid_mpyacc(a1, k0, v1);
            b1 = lid_mpyacc(b1, k1, v1);
            a2 = lid_mpyacc(a2, k0, v2);
            b2 = lid_mpyacc(b2, k1, v2);
            a3 = lid_mpyacc(a3, k0, v3);
            b3 = lid_mpyacc(b3, k1, v3);
        }

        out0 = lid_head_acc(out0, a0, w_t[h + 0]);
        out0 = lid_head_acc(out0, a1, w_t[h + 1]);
        out0 = lid_head_acc(out0, a2, w_t[h + 2]);
        out0 = lid_head_acc(out0, a3, w_t[h + 3]);
        out1 = lid_head_acc(out1, b0, w_t[h + 0]);
        out1 = lid_head_acc(out1, b1, w_t[h + 1]);
        out1 = lid_head_acc(out1, b2, w_t[h + 2]);
        out1 = lid_head_acc(out1, b3, w_t[h + 3]);
    }
    for (; h < n_head; ++h) {
        const HVX_Vector * restrict q0 = q_t + h * D;

        HVX_VectorPair a0 = Q6_W_vzero();
        HVX_VectorPair b0 = Q6_W_vzero();
        for (uint32_t d = 0; d < D; ++d) {
            const HVX_Vector v0 = q0[d];
            a0 = lid_mpyacc(a0, kt0[d], v0);
            b0 = lid_mpyacc(b0, kt1[d], v0);
        }
        out0 = lid_head_acc(out0, a0, w_t[h]);
        out1 = lid_head_acc(out1, b0, w_t[h]);
    }

    lid_store_row(o0, out0, m0);
    lid_store_row(o1, out1, m1);
}

// Each thread owns a range of kv blocks and streams K / mask in, output out (HTP_LID_N_BUF buffers)
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

    const bool     is_f32     = (k->type == HTP_TYPE_F32);
    const bool     is_q8_0    = (k->type == HTP_TYPE_Q8_0);
    const bool     is_q4_0    = (k->type == HTP_TYPE_Q4_0);
    const uint32_t k_row_size = lctx->kp->k_row_size;

    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbm1 = m->nb[1];
    const uint32_t nb1  = dst->nb[1];

    const dma_addr_t k_base = k->data   + lctx->is  * k->nb[3];
    const dma_addr_t m_base = m->data   + lctx->t0  * nbm1 + lctx->im3 * m->nb[3];
    const dma_addr_t d_base = dst->data + lctx->t0  * nb1  + lctx->is  * dst->nb[3];

    uint8_t * spad_k = lctx->spad_k + ith * HTP_LID_N_BUF * L->size_k_buf;
    uint8_t * spad_m = lctx->spad_m + ith * HTP_LID_N_BUF * L->size_m_buf;
    uint8_t * spad_o = lctx->spad_o + ith * HTP_LID_N_BUF * L->size_o_buf;

    HVX_Vector * kt[2];
    kt[0] = (HVX_Vector *) (lctx->spad_kt + ith * L->size_kt);
    kt[1] = kt[0] + hex_round_up(D, HTP_LID_KV_BLOCK);
    HVX_Vector * kt_t0 = kt[1] + hex_round_up(D, HTP_LID_KV_BLOCK);
    HVX_Vector * kt_t1 = kt_t0 + HTP_LID_KV_BLOCK;

    // Q8_0: transpose by gather from the raw rows (sc: one vector of block scales per 32 d)
    const bool       q8_g   = is_q8_0 && lid_q8_0_gather_ok(L->size_k_raw_row, D);
    HVX_Vector *     sc     = kt_t1 + HTP_LID_KV_BLOCK;
    const HVX_Vector g_offs = hvx_vec_gather_offsets_h(L->size_k_raw_row);

    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];
    dma_queue *               dma_q = octx->ctx->dma[ith];

    for (uint32_t b = b0, i = 0; b < b1 && i < HTP_LID_N_BUF; ++b, ++i) {
        const uint32_t kv0 = b * HTP_LID_KV_BLOCK;
        const uint32_t nkv = MIN(HTP_LID_KV_BLOCK, n_kv - kv0);

        dma_queue_push(dma_q, dma_make_data(d_base, spad_o + i * L->size_o_buf),
                       nb1, L->size_o_row, nkv * sizeof(float), 0);
        dma_queue_push(dma_q, dma_make_data(spad_k + i * L->size_k_buf, k_base + kv0 * nbk2),
                       L->size_k_raw_row, nbk2, k_row_size, nkv);
        dma_queue_push(dma_q, dma_make_data(spad_m + i * L->size_m_buf, m_base + kv0 * sizeof(__fp16)),
                       L->size_m_row, nbm1, nkv * sizeof(__fp16), nt);
    }

    // Two kv blocks per step, the next two are in flight
    for (uint32_t b = b0; b < b1; b += 2) {
        const uint32_t nb = MIN(2, b1 - b);

        uint8_t * o_buf[2];
        uint8_t * k_buf[2];
        uint8_t * m_buf[2];

        for (uint32_t j = 0; j < nb; ++j) {
            const uint32_t nkv = MIN(HTP_LID_KV_BLOCK, n_kv - (b + j) * HTP_LID_KV_BLOCK);

            o_buf[j] = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;
            k_buf[j] = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;
            m_buf[j] = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;

            if ((is_q8_0 && !q8_g) || is_q4_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) (b + j));
                lid_dequant_block(k_buf[j], L->size_k_raw_row, nkv, D, is_q4_0);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, (uint16_t) (b + j));
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) (b + j));
            if (is_f32) {
                for (uint32_t r = 0; r < nkv; ++r) {
                    lid_convert_row_f32_to_f16(k_buf[j] + r * L->size_k_raw_row, D);
                }
            }
            if (q8_g) {
                lid_transpose_q8_0_gather(kt[j], kt_t0, sc, k_buf[j], L->size_k_raw_row, g_offs, D);
            } else {
                for (uint32_t d0 = 0; d0 < D; d0 += HTP_LID_KV_BLOCK) {
                    lid_transpose_64x64_f16(kt[j] + d0, k_buf[j] + d0 * sizeof(__fp16), L->size_k_raw_row, nkv, kt_t0, kt_t1);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) (b + j));
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);
        for (uint32_t t = 0; t < nt; ++t) {
            const HVX_Vector * q_t = (const HVX_Vector *) lctx->spad_q + t * n_head * D;
            const HVX_Vector * w_t = (const HVX_Vector *) (lctx->spad_w + t * n_head * VLEN);
            if (nb == 2) {
                lid_compute_row2((float *) (o_buf[0] + t * L->size_o_row), (float *) (o_buf[1] + t * L->size_o_row),
                                 q_t, w_t, kt[0], kt[1],
                                 (const __fp16 *) (m_buf[0] + t * L->size_m_row), (const __fp16 *) (m_buf[1] + t * L->size_m_row),
                                 D, n_head);
            } else {
                lid_compute_row((float *) (o_buf[0] + t * L->size_o_row), q_t, w_t, kt[0],
                                (const __fp16 *) (m_buf[0] + t * L->size_m_row), D, n_head);
            }
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) b);

        for (uint32_t j = 0; j < nb; ++j) {
            const uint32_t kv0 = (b + j) * HTP_LID_KV_BLOCK;
            const uint32_t nkv = MIN(HTP_LID_KV_BLOCK, n_kv - kv0);

            dma_queue_push(dma_q, dma_make_data(d_base + kv0 * sizeof(float), o_buf[j]),
                           nb1, L->size_o_row, nkv * sizeof(float), nt);

            if (b + j + HTP_LID_N_BUF < b1) {
                const uint32_t kv2  = kv0 + HTP_LID_N_BUF * HTP_LID_KV_BLOCK;
                const uint32_t nkv2 = MIN(HTP_LID_KV_BLOCK, n_kv - kv2);

                dma_queue_push(dma_q, dma_make_data(k_buf[j], k_base + kv2 * nbk2),
                               L->size_k_raw_row, nbk2, k_row_size, nkv2);
                dma_queue_push(dma_q, dma_make_data(m_buf[j], m_base + kv2 * sizeof(__fp16)),
                               L->size_m_row, nbm1, nkv2 * sizeof(__fp16), nt);
            }
        }
    }

    for (uint32_t i = 0; i < MIN(b1 - b0, HTP_LID_N_BUF); ++i) {
        dma_queue_pop(dma_q);
    }

    (void) nth;
}

// ----- HMX path: K is the activation, Q the weights, head sum by a second HMX pass -----
//
// Channel c = t * n_head + h of a token tile. Per kv row tile (32 kv) and token tile u (32 tokens):
//   mm1: S[kv, c] = relu(k . q_c) * |w_c|  (convert scale |w_c|, pos clamp)
//   mm2: O[kv, t] = sum_c S[kv, c] * G[c, t], G[c, t] = sign(w_c) for c of token t
// O tiles hold f16 (kv pair, token) words; gather one token column, add the mask, store f32.
//
// A run preps the Q sets of up to n_qsets (stream, token tile) pairs, then pipelines all their kv rows.
// Row x of a run is kv x % kv_seg of set x / kv_seg (kv_seg = n_kv padded to 64), so a block of Bc rows can
// span sets: it is split in segments, one per set. HMX computes block g while HVX preps K of g + 1 and extracts O of g - 1.

struct lid_hmx_set {
    uint32_t is;
    uint32_t im3;
    uint32_t t0;
    uint32_t nt;
    uint32_t n_ch;
    uint32_t n_u;
};

struct htp_lid_hmx_context {
    const struct htp_ops_context *       octx;
    const struct htp_lid_kernel_params * kp;
    struct htp_lid_hmx_vtcm_layout       L;

    uint32_t Bc;
    uint32_t kv_seg;    // rows per set: n_kv padded to 64
    uint32_t n_blocks;  // per run
    uint32_t n_tile;    // tokens per token tile
    uint32_t n_tt;      // token tiles per stream

    uint32_t n_qsets;   // Q sets per buffer
    uint32_t qbuf;      // Q buffer of the run (two buffers: the next run's Q is prepped during this run)

    uint32_t tile0;     // first (stream, token tile) of the run
    uint32_t n_run;     // sets in the run
    uint32_t n_qthr;    // threads with Q work in the run
    uint32_t nt_run;    // tokens of each set in the run
    uint32_t n_u_run;   // token tiles of 32 of each set in the run
    int32_t  g;         // current stage

    struct lid_hmx_set sets[HTP_LID_HMX_MAX_QSETS];

    // next run, Q prepped in the stages of this one
    bool                  nx_on;
    uint32_t              nx_n_run;
    struct fastdiv_values nx_stg_div; // stages of this run: n_blocks + 1
    struct lid_hmx_set    nx_sets[HTP_LID_HMX_MAX_QSETS];

    __fp16 *  qw;
    __fp16 *  gw;
    uint8_t * sc;
    uint8_t * sc_id;
    __fp16 *  s;
    float *   w;
    uint8_t * qs;
    uint8_t * k_raw[2];
    __fp16 *  k_t[2];
    __fp16 *  o[2];
    uint8_t * m[4];
    uint8_t * out[2];
};

#define LID_HMX_MAX_SEGS HTP_LID_HMX_MAX_QSETS

// Rows of one set in a block: valid rows [x0, x0 + nv) of the block are kv [kv0, kv0 + nv) of the set
struct lid_hmx_seg {
    uint32_t set;
    uint32_t x0;
    uint32_t kv0;
    uint32_t nv;
};

static inline struct lid_hmx_set lid_hmx_set_init(const struct htp_lid_hmx_context * hctx, uint32_t set) {
    const struct htp_ops_context * octx = hctx->octx;
    const uint32_t ti = hctx->tile0 + set;

    struct lid_hmx_set S;
    S.is   = fastdiv(ti, &hctx->kp->div_n_tt);
    S.im3  = fastmodulo(S.is, octx->src[3]->ne[3], &hctx->kp->div_m_ne3);
    S.t0   = (ti - S.is * hctx->n_tt) * hctx->n_tile;
    S.nt   = MIN(hctx->n_tile, octx->src[0]->ne[2] - S.t0);
    S.n_ch = (S.nt * octx->src[0]->ne[1] + 31) / 32;
    S.n_u  = (S.nt + 31) / 32;
    return S;
}

// Segments of block g with valid rows. Returns their count, n_rows gets the block rows (multiple of 64).
static inline uint32_t lid_hmx_segs(const struct htp_lid_hmx_context * hctx, uint32_t g, struct lid_hmx_seg * seg, uint32_t * n_rows) {
    const uint32_t n_kv = hctx->octx->src[1]->ne[2];
    const uint32_t ks   = hctx->kv_seg;
    const uint32_t xa   = g * hctx->Bc;
    const uint32_t xb   = MIN(xa + hctx->Bc, hctx->n_run * ks);

    uint32_t n = 0;
    for (uint32_t x = xa; x < xb;) {
        const uint32_t set = fastdiv(x, &hctx->kp->div_kv_seg);
        const uint32_t kv  = x - set * ks;
        const uint32_t end = MIN(xb, (set + 1) * ks);
        if (kv < n_kv) {
            seg[n].set = set;
            seg[n].x0  = x - xa;
            seg[n].kv0 = kv;
            seg[n].nv  = MIN(n_kv, kv + (end - x)) - kv;
            n++;
        }
        x = end;
    }
    *n_rows = xb - xa;
    return n;
}

typedef struct {
    const __fp16 *  qw;
    const __fp16 *  gw;
    const uint8_t * sc;
    uint32_t        rt0;
    uint32_t        n_rt;
} lid_hmx_job_seg_t;

typedef struct {
    lid_hmx_job_seg_t seg[LID_HMX_MAX_SEGS];
    uint32_t          n_seg;

    const __fp16 *  k_t;
    const uint8_t * sc_id;
    __fp16 *        s;
    __fp16 *        o;
    uint32_t        n_ch;
    uint32_t        n_u;
    uint32_t        n_dot;
    uint32_t        n_head;
    uint32_t        n_sg;
} lid_hmx_job_t;

// acc += sum of n tile products, a and b advance one tile per step
static inline void lid_hmx_dot(const __fp16 * a, const __fp16 * b, uint32_t n) {
    for (; n > 32; n -= 32) {
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :: "r"(65535), "r"(a), "r"(b));
        a += 32 * HMX_FP16_TILE_N_ELMS;
        b += 32 * HMX_FP16_TILE_N_ELMS;
    }
    asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :: "r"(n * HMX_FP16_TILE_SIZE - 1), "r"(a), "r"(b));
}

// Per segment: channel tile outer, row tile inner, one bias load per channel tile. S holds [n_rt][n_sg] tiles.
static void lid_hmx_worker(void * data) {
    const lid_hmx_job_t * job = (const lid_hmx_job_t *) data;

    const size_t   dot_stride = (size_t) job->n_dot * HMX_FP16_TILE_N_ELMS;
    const size_t   s_stride   = (size_t) job->n_sg * HMX_FP16_TILE_N_ELMS;
    const uint32_t nh         = job->n_head;

    for (uint32_t i = 0; i < job->n_seg; ++i) {
        const lid_hmx_job_seg_t * sg   = &job->seg[i];
        const uint32_t            n_rt = sg->n_rt;
        const __fp16 *            kt   = job->k_t + sg->rt0 * dot_stride;
        __fp16 *                  ot   = job->o + sg->rt0 * job->n_u * HMX_FP16_TILE_N_ELMS;

        for (uint32_t u = 0; u < job->n_u; ++u) {
            const uint32_t j0 = u * nh;
            const uint32_t j1 = MIN(j0 + nh, job->n_ch);

            for (uint32_t c = j0; c < j1; ++c) {
                const __fp16 * qw = sg->qw + c * dot_stride;
                __fp16 *       st = job->s + (c - j0) * HMX_FP16_TILE_N_ELMS;

                asm volatile(HMX_SET_BIAS("%0") :: "r"(sg->sc + c * 256));
                for (uint32_t rt = 0; rt < n_rt; ++rt) {
                    asm volatile(HMX_CLRACC_F16());
                    lid_hmx_dot(kt + rt * dot_stride, qw, job->n_dot);
                    asm volatile("mxmem(%0, %1):after:pos.hf = acc\n" :: "r"(st + rt * s_stride), "r"(0) : "memory");
                }
            }

            asm volatile(HMX_SET_BIAS("%0") :: "r"(job->sc_id));
            for (uint32_t rt = 0; rt < n_rt; ++rt) {
                asm volatile(HMX_CLRACC_F16());
                lid_hmx_dot(job->s + rt * s_stride, sg->gw + j0 * HMX_FP16_TILE_N_ELMS, j1 - j0);
                asm volatile(HMX_STORE_AFTER_F16("%0", "%1") :: "r"(ot + (rt * job->n_u + u) * HMX_FP16_TILE_N_ELMS), "r"(0) : "memory");
            }
        }
    }
}

// Thread slices: K rows by row pairs of the padded block, mask / output rows by token.
// Block 0 is prepped together with Q: its K rows go to the threads without Q work.
// K rows are split only in stages on all n_threads threads.
static inline void lid_hmx_rows(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g, uint32_t n_rows,
                                uint32_t * r0, uint32_t * r1) {
    uint32_t per;
    if (g == 0 && hctx->n_qthr < nth) {
        if (ith < hctx->n_qthr) {
            *r0 = *r1 = 0;
            return;
        }
        ith -= hctx->n_qthr;
        per = (n_rows + nth - hctx->n_qthr - 1) / (nth - hctx->n_qthr);
    } else {
        per = fastdiv(n_rows + hctx->octx->n_threads - 1, &hctx->octx->n_threads_div);
    }
    per = hex_round_up(per, 2);
    *r0 = MIN(ith * per, n_rows);
    *r1 = MIN(*r0 + per, n_rows);
}

// Split over n_threads; a stage on one thread has one token, which this split also gives to thread 0
static inline void lid_hmx_toks(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nt, uint32_t * t0, uint32_t * t1) {
    const uint32_t per = fastdiv(nt + hctx->octx->n_threads - 1, &hctx->octx->n_threads_div);
    *t0 = MIN(ith * per, nt);
    *t1 = MIN((ith + 1) * per, nt);
}

// Push (or only count, push == false) the K row and mask DMAs of block g for this thread, one per segment
static uint32_t lid_hmx_in(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g, bool push) {
    const struct htp_ops_context * octx = hctx->octx;
    const struct htp_tensor *      k    = octx->src[1];
    const struct htp_tensor *      m    = octx->src[3];
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;

    struct lid_hmx_seg seg[LID_HMX_MAX_SEGS];
    uint32_t           n_rows;
    const uint32_t     n_seg = lid_hmx_segs(hctx, g, seg, &n_rows);

    uint32_t r0, r1, ta, tb;
    lid_hmx_rows(hctx, ith, nth, g, n_rows, &r0, &r1);
    lid_hmx_toks(hctx, ith, hctx->nt_run, &ta, &tb);

    dma_queue *    dma_q      = octx->ctx->dma[ith];
    const uint32_t k_row_size = hctx->kp->k_row_size;

    uint32_t n = 0;
    for (uint32_t i = 0; i < n_seg; ++i) {
        const struct lid_hmx_set S = hctx->sets[seg[i].set];

        const uint32_t ra = MAX(r0, seg[i].x0);
        const uint32_t rb = MIN(r1, seg[i].x0 + seg[i].nv);
        if (ra < rb) {
            if (push) {
                dma_queue_push(dma_q, dma_make_data(hctx->k_raw[g % 2] + ra * L->size_k_raw_row,
                                                    k->data + S.is * k->nb[3] + (seg[i].kv0 + ra - seg[i].x0) * k->nb[2]),
                               L->size_k_raw_row, k->nb[2], k_row_size, rb - ra);
            }
            n++;
        }
        if (ta < tb) {
            if (push) {
                dma_queue_push(dma_q, dma_make_data(hctx->m[g % 4] + ta * L->size_m_row + seg[i].x0 * sizeof(__fp16),
                                                    m->data + S.im3 * m->nb[3] + (S.t0 + ta) * m->nb[1] + seg[i].kv0 * sizeof(__fp16)),
                               L->size_m_row, m->nb[1], seg[i].nv * sizeof(__fp16), tb - ta);
            }
            n++;
        }
    }
    return n;
}

static inline uint32_t lid_hmx_push_in(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g) {
    return lid_hmx_in(hctx, ith, nth, g, true);
}

// Row pair (a, b) quants of two blocks (64 values each, bytes 0..63) to the two tile vectors, scaled by s0, s1
static inline void lid_q8_0_pair_store(HVX_Vector * restrict dst, HVX_Vector a, HVX_Vector b, HVX_Vector s0, HVX_Vector s1, bool two) {
    const HVX_VectorPair qh = Q6_Wh_vunpack_Vb(Q6_V_lo_W(Q6_W_vshuff_VVR(b, a, -1)));
    dst[0] = LID_MUL_F16(Q6_Vhf_equals_Vh(Q6_V_lo_W(qh)), s0);
    if (two) {
        dst[16] = LID_MUL_F16(Q6_Vhf_equals_Vh(Q6_V_hi_W(qh)), s1);
    }
}

// Raw Q8_0 rows [r0, r1) to activation tiles, dequant fused with the row pair interleave. Rows >= nv are zero.
// Per row pair and two blocks: interleave the quants of both rows as bytes, widen, then scale by the (d_r, d_r+1) word.
static inline void lid_hmx_prep_k_q8_0(__fp16 * restrict k_t, const uint8_t * restrict k_raw, size_t row_stride,
                                       uint32_t r0, uint32_t r1, uint32_t nv, uint32_t n_dot) {
    const HVX_Vector     zero = Q6_V_vzero();
    const HVX_VectorPred q32  = Q6_Q_vsetq_R(32);

    for (uint32_t r = r0; r < r1; r += 2) {
        const uint8_t * p0 = k_raw + r * row_stride;
        const uint8_t * p1 = p0 + row_stride;
        const bool      v0 = r < nv;
        const bool      v1 = r + 1 < nv;

        HVX_Vector * dst = (HVX_Vector *) (k_t + (r / 32) * n_dot * HMX_FP16_TILE_N_ELMS) + (r % 32) / 2;

        if (n_dot == 4) {
            // D = 128: the row is 136 bytes, two aligned loads; scales at halfwords 0, 17, 34, 51
            const HVX_Vector a0 = v0 ? ((const HVX_Vector *) p0)[0] : zero;
            const HVX_Vector a1 = v0 ? ((const HVX_Vector *) p0)[1] : zero;
            const HVX_Vector b0 = v1 ? ((const HVX_Vector *) p1)[0] : zero;
            const HVX_Vector b1 = v1 ? ((const HVX_Vector *) p1)[1] : zero;

            const HVX_VectorPair ds = Q6_W_vshuff_VVR(b0, a0, -SIZEOF_FP16);

            lid_q8_0_pair_store(dst,
                                Q6_V_vmux_QVV(q32, Q6_V_valign_VVR(a0, a0, 2), Q6_V_valign_VVR(a0, a0, 4)),
                                Q6_V_vmux_QVV(q32, Q6_V_valign_VVR(b0, b0, 2), Q6_V_valign_VVR(b0, b0, 4)),
                                hvx_vec_repl4(Q6_V_lo_W(ds)), hvx_vec_repl4(Q6_V_vror_VR(Q6_V_lo_W(ds), 17 * 4)), true);
            lid_q8_0_pair_store(dst + 32,
                                Q6_V_vmux_QVV(q32, Q6_V_valign_VVR(a1, a0, 70), Q6_V_valign_VVR(a1, a0, 72)),
                                Q6_V_vmux_QVV(q32, Q6_V_valign_VVR(b1, b0, 70), Q6_V_valign_VVR(b1, b0, 72)),
                                hvx_vec_repl4(Q6_V_vror_VR(Q6_V_hi_W(ds), 2 * 4)), hvx_vec_repl4(Q6_V_vror_VR(Q6_V_hi_W(ds), 19 * 4)), true);
            continue;
        }

        for (uint32_t i = 0; i < n_dot; i += 2) {
            const uint8_t * pa = p0 + i * sizeof(block_q8_0);
            const uint8_t * pb = p1 + i * sizeof(block_q8_0);

            // quants of blocks i, i + 1 in bytes 0..63, scales d_i at byte 0 and d_i+1 at byte 34
            const HVX_Vector a  = v0 ? Q6_V_vmux_QVV(q32, *(const HVX_UVector *) (pa + 2), *(const HVX_UVector *) (pa + 4)) : zero;
            const HVX_Vector b  = v1 ? Q6_V_vmux_QVV(q32, *(const HVX_UVector *) (pb + 2), *(const HVX_UVector *) (pb + 4)) : zero;
            const HVX_Vector da = v0 ? *(const HVX_UVector *) pa : zero;
            const HVX_Vector db = v1 ? *(const HVX_UVector *) pb : zero;
            const HVX_Vector ds = Q6_V_lo_W(Q6_W_vshuff_VVR(db, da, -SIZEOF_FP16));

            lid_q8_0_pair_store(dst + i * 16, a, b, hvx_vec_repl4(ds), hvx_vec_repl4(Q6_V_vror_VR(ds, 2 * sizeof(block_q8_0))), i + 1 < n_dot);
        }
    }
}

// Raw Q4_0 rows [r0, r1) to activation tiles, same as Q8_0: per row pair and two blocks, one load per row.
// Rows >= nv are zero.
static inline void lid_hmx_prep_k_q4_0(__fp16 * restrict k_t, const uint8_t * restrict k_raw, size_t row_stride,
                                       uint32_t r0, uint32_t r1, uint32_t nv, uint32_t n_dot) {
    const HVX_Vector     zero = Q6_V_vzero();
    const HVX_VectorPred q32  = Q6_Q_vsetq_R(32);
    const HVX_VectorPred lo16 = Q6_Q_or_QQ(Q6_Q_vsetq_R(16), Q6_Q_and_QQn(Q6_Q_vsetq_R(48), q32));

    for (uint32_t r = r0; r < r1; r += 2) {
        const uint8_t * p0 = k_raw + r * row_stride;
        const uint8_t * p1 = p0 + row_stride;
        const bool      v0 = r < nv;
        const bool      v1 = r + 1 < nv;

        HVX_Vector * dst = (HVX_Vector *) (k_t + (r / 32) * n_dot * HMX_FP16_TILE_N_ELMS) + (r % 32) / 2;

        for (uint32_t i = 0; i < n_dot; i += 2) {
            // blocks i, i + 1: scales d_i at byte 0 and d_i+1 at byte 18
            const HVX_Vector da = v0 ? *(const HVX_UVector *) (p0 + i * sizeof(block_q4_0)) : zero;
            const HVX_Vector db = v1 ? *(const HVX_UVector *) (p1 + i * sizeof(block_q4_0)) : zero;
            const HVX_Vector ds = Q6_V_lo_W(Q6_W_vshuff_VVR(db, da, -SIZEOF_FP16));

            lid_q8_0_pair_store(dst + i * 16, lid_q4_0_pair_quants(da, q32, lo16), lid_q4_0_pair_quants(db, q32, lo16),
                                hvx_vec_repl4(ds), hvx_vec_repl4(Q6_V_vror_VR(ds, 2 * sizeof(block_q4_0))), i + 1 < n_dot);
        }
    }
}

// F16 rows [r0, r1) to activation tiles: row pairs interleaved. Rows >= nv are zero.
static inline void lid_hmx_prep_k_f16(__fp16 * restrict k_t, const uint8_t * restrict k_raw, size_t row_stride,
                                      uint32_t r0, uint32_t r1, uint32_t nv, uint32_t n_dot) {
    const HVX_Vector zero = Q6_V_vzero();

    for (uint32_t r = r0; r < r1; r += 2) {
        const HVX_Vector * p0 = (const HVX_Vector *) (k_raw + r * row_stride);
        const HVX_Vector * p1 = (const HVX_Vector *) (k_raw + (r + 1) * row_stride);
        const bool         v0 = r < nv;
        const bool         v1 = r + 1 < nv;

        HVX_Vector * dst = (HVX_Vector *) (k_t + (r / 32) * n_dot * HMX_FP16_TILE_N_ELMS) + (r % 32) / 2;

        for (uint32_t i = 0; i < n_dot; i += 2) {
            const HVX_VectorPair p = Q6_W_vshuff_VVR(v1 ? p1[i / 2] : zero, v0 ? p0[i / 2] : zero, -SIZEOF_FP16);
            dst[i * 16] = Q6_V_lo_W(p);
            if (i + 1 < n_dot) {
                dst[(i + 1) * 16] = Q6_V_hi_W(p);
            }
        }
    }
}

// K rows of block g (this thread's rows) to activation tiles, one piece per set; set padding rows are zero.
// Not inlined: keeps the stage's registers out of the prep loops.
static __attribute__((noinline)) void lid_hmx_prep_k(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g) {
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;
    const struct htp_tensor * k = hctx->octx->src[1];

    const uint32_t n_kv   = k->ne[2];
    const uint32_t ks     = hctx->kv_seg;
    const uint32_t xa     = g * hctx->Bc;
    const uint32_t n_rows = MIN(hctx->Bc, hctx->n_run * ks - xa);

    uint32_t r0, r1;
    lid_hmx_rows(hctx, ith, nth, g, n_rows, &r0, &r1);

    uint8_t *        k_raw = hctx->k_raw[g % 2];
    __fp16 *        k_t   = hctx->k_t[g % 2];
    const bool      f32   = k->type == HTP_TYPE_F32;
    const bool      q8    = k->type == HTP_TYPE_Q8_0;
    const bool      q4    = k->type == HTP_TYPE_Q4_0;

    for (uint32_t r = r0; r < r1;) {
        const uint32_t x   = xa + r;
        const uint32_t set = fastdiv(x, &hctx->kp->div_kv_seg);
        const uint32_t kv  = x - set * ks;
        const uint32_t end = MIN(r1, (set + 1) * ks - xa);
        const uint32_t nv  = r + (kv < n_kv ? n_kv - kv : 0);

        if (q8) {
            lid_hmx_prep_k_q8_0(k_t, k_raw, L->size_k_raw_row, r, end, nv, L->n_dot);
        } else if (q4) {
            lid_hmx_prep_k_q4_0(k_t, k_raw, L->size_k_raw_row, r, end, nv, L->n_dot);
        } else {
            if (f32) {
                for (uint32_t i = r; i < end; ++i) {
                    lid_convert_row_f32_to_f16(k_raw + i * L->size_k_raw_row, k->ne[0]);
                }
            }
            lid_hmx_prep_k_f16(k_t, k_raw, L->size_k_raw_row, r, end, nv, L->n_dot);
        }
        r = end;
    }
}

// O column of each token (64 kv per gather) + mask -> f32 rows, then one DMA out per segment
static void lid_hmx_extract(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g) {
    const struct htp_ops_context * octx = hctx->octx;
    const struct htp_tensor *      dst  = octx->dst;
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;

    uint32_t ta, tb;
    lid_hmx_toks(hctx, ith, hctx->nt_run, &ta, &tb);
    if (ta >= tb) {
        return;
    }

    struct lid_hmx_seg seg[LID_HMX_MAX_SEGS];
    uint32_t           n_rows;
    const uint32_t     n_seg = lid_hmx_segs(hctx, g, seg, &n_rows);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) g);

    const uint32_t n_u = hctx->n_u_run;
    const uint32_t mu  = (n_u + 1) * HMX_FP16_TILE_SIZE - 1;

    int32_t offs[32] __attribute__((aligned(128)));
    for (int l = 0; l < 32; l++) {
        offs[l] = (l / 16) * n_u * HMX_FP16_TILE_SIZE + (l % 16) * 128;
    }
    const HVX_Vector v_offs = *(const HVX_Vector *) offs;

    HVX_Vector * gbuf = (HVX_Vector *) (hctx->qs + ith * L->size_qs + 32 * (L->size_qs_row + L->size_qh_row));

    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector ninf = Q6_V_vsplat_R(0xFF800000);
    const HVX_Vector one  = Q6_Vh_vsplat_R(0x3C00);
    const HVX_Vector hinf = Q6_Vh_vsplat_R(0xFC00);

    const __fp16 *  o_base = hctx->o[g % 2];
    const uint8_t * m_buf  = hctx->m[g % 4];
    uint8_t *       o_buf  = hctx->out[g % 2];

    for (uint32_t i = 0; i < n_seg; ++i) {
        const uint32_t q0 = seg[i].x0 / 64;
        const uint32_t q1 = (seg[i].x0 + seg[i].nv + 63) / 64;

        for (uint32_t t = ta; t < tb; ++t) {
            const uint32_t   u   = t / 32;
            const HVX_Vector off = Q6_Vw_vadd_VwVw(v_offs, Q6_V_vsplat_R((t % 32) * 4));

            for (uint32_t q = q0; q < q1; ++q) {
                const __fp16 * base = o_base + (2 * q * n_u + u) * HMX_FP16_TILE_N_ELMS;
                Q6_vgather_ARMVw(gbuf + q - q0, (size_t) base, mu, off);
            }
            hvx_gather_sync(gbuf + q1 - q0 - 1);

            const HVX_Vector * mv = (const HVX_Vector *) (m_buf + t * L->size_m_row);
            HVX_Vector *       ov = (HVX_Vector *) (o_buf + t * L->size_o_row);

            for (uint32_t q = q0; q < q1; ++q) {
                const HVX_VectorPair s = hvx_vec_f16_to_f32(gbuf[q - q0]);

                // qf32 math does not keep inf: add a finite mask, then put -inf back
                const HVX_Vector     m_f16 = mv[q];
                const HVX_VectorPred m_inf = Q6_Q_vcmp_eq_VhVh(m_f16, hinf);
                const HVX_VectorPair m_f32 = hvx_vec_f16_to_f32(Q6_V_vmux_QVV(m_inf, zero, m_f16));
                const HVX_VectorPair f_f32 = hvx_vec_f16_to_f32(Q6_V_vmux_QVV(m_inf, one, zero));

                const HVX_Vector acc0 = HVX_OP_ADD_F32(Q6_V_lo_W(s), Q6_V_lo_W(m_f32));
                const HVX_Vector acc1 = HVX_OP_ADD_F32(Q6_V_hi_W(s), Q6_V_hi_W(m_f32));
                ov[2 * q + 0] = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_lo_W(f_f32), zero), ninf, acc0);
                ov[2 * q + 1] = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_hi_W(f_f32), zero), ninf, acc1);
            }
        }

        const struct lid_hmx_set S = hctx->sets[seg[i].set];
        dma_queue_push(octx->ctx->dma[ith],
                       dma_make_data(dst->data + S.is * dst->nb[3] + (S.t0 + ta) * dst->nb[1] + seg[i].kv0 * sizeof(float),
                                     o_buf + ta * L->size_o_row + seg[i].x0 * sizeof(float)),
                       dst->nb[1], L->size_o_row, seg[i].nv * sizeof(float), tb - ta);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) g);
}

// Number of output DMAs of block g for this thread
static inline uint32_t lid_hmx_n_out(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth, uint32_t g) {
    uint32_t ta, tb;
    lid_hmx_toks(hctx, ith, hctx->nt_run, &ta, &tb);
    if (ta >= tb) {
        return 0;
    }
    struct lid_hmx_seg seg[LID_HMX_MAX_SEGS];
    uint32_t           n_rows;
    return lid_hmx_segs(hctx, g, seg, &n_rows);
}

static void lid_hmx_q_chunk(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth);

// Pipeline stage g (HMX runs block g meanwhile): prep block g + 1, extract block g - 1
static void lid_hmx_stage(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth) {
    const int32_t g = hctx->g;
    const int32_t n = (int32_t) hctx->n_blocks;

    struct htp_thread_trace * tr    = &hctx->octx->ctx->trace[ith];
    dma_queue *               dma_q = hctx->octx->ctx->dma[ith];

    // FIFO order per thread: OUT(g - 3), IN(g + 1), OUT(g - 2). When this stage preps next run Q, also pop
    // OUT(g - 2) so the queue is empty for the Q DMAs; else leave it in flight for one more stage.
    const int32_t lag = hctx->nx_on ? 2 : 3;

    uint32_t n_pop = 0;
    if (lag == 3 && g - 3 >= 0) {
        n_pop += lid_hmx_n_out(hctx, ith, nth, g - 3);
    }
    if (g >= 0 && g + 1 < n) {
        n_pop += lid_hmx_in(hctx, ith, nth, g + 1, false);
    }
    if (lag == 2 && g - 2 >= 0) {
        n_pop += lid_hmx_n_out(hctx, ith, nth, g - 2);
    }
    for (uint32_t i = 0; i < n_pop; ++i) {
        dma_queue_pop(dma_q);
    }

    if (hctx->nx_on && g >= 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) g);
        lid_hmx_q_chunk(hctx, ith, nth);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) g);
    }
    if (g + 2 < n) {
        lid_hmx_push_in(hctx, ith, nth, g + 2);
    }
    if (g + 1 < n) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) (g + 1));
        lid_hmx_prep_k(hctx, ith, nth, g + 1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) (g + 1));
    }
    if (g - 1 >= 0) {
        lid_hmx_extract(hctx, ith, nth, g - 1);
    }
    if (g == n) {
        uint32_t n_tail = lid_hmx_n_out(hctx, ith, nth, n - 1);
        if (lag == 3 && n >= 2) {
            n_tail += lid_hmx_n_out(hctx, ith, nth, n - 2);
        }
        for (uint32_t i = 0; i < n_tail; ++i) {
            dma_queue_pop(dma_q);
        }
    }
}

static void lid_hmx_stage_thread(unsigned int nth, unsigned int ith, void * data) {
    lid_hmx_stage((const struct htp_lid_hmx_context *) data, ith, nth);
}

// Push the Q rows of channel tile c of set S, one DMA per token. Returns the number of DMA descriptors.
static uint32_t lid_hmx_push_q(const struct htp_lid_hmx_context * hctx, dma_queue * dma_q, uint8_t * qs,
                               const struct lid_hmx_set * S, uint32_t c) {
    const struct htp_tensor * q  = hctx->octx->src[0];
    const uint32_t            nh = q->ne[1];
    const uint32_t            ch = c * 32;
    const uint32_t            nr = MIN(32, S->nt * nh - ch);

    // channel ch + r is (token t, head h); a DMA ends at the token end, so the next one starts at h 0
    uint32_t t = fastdiv(ch, &hctx->kp->div_n_head);
    uint32_t h = ch - t * nh;
    uint32_t n = 0;
    for (uint32_t r = 0; r < nr; ++t, h = 0) {
        const uint32_t cnt = MIN(nh - h, nr - r);
        dma_queue_push(dma_q, dma_make_data(qs + r * hctx->L.size_qs_row,
                                            q->data + S->is * q->nb[3] + (S->t0 + t) * q->nb[2] + h * q->nb[1]),
                       hctx->L.size_qs_row, q->nb[1], q->ne[0] * sizeof(float), cnt);
        r += cnt;
        n++;
    }
    return n;
}

// Scales (|w| per channel, f16 in the low half of each word) and G tile (sign(w) at row c, column token(c) % 32)
// of one channel tile. w points at the tile's 32 channels, the first one is (token t, head h).
static inline void lid_hmx_build_sc_g(uint8_t * restrict sc, __fp16 * restrict gt, const float * restrict w,
                                      uint32_t t, uint32_t h, uint32_t nr, uint32_t nh) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector wv   = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(nr * sizeof(float)), *(const HVX_Vector *) w, zero);

    HVX_Vector * scv = (HVX_Vector *) sc;
    scv[0] = hvx_vec_f32_to_f16_shuff(Q6_V_vand_VV(wv, Q6_V_vsplat_R(0x7FFFFFFF)), zero);
    scv[1] = zero;

    // sign(w) as f16 bits in the low half of each word, then packed: row r in halfword r
    const HVX_VectorPred nz  = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(wv, Q6_V_vsplat_R(0x7FFFFFFF)), zero);
    const HVX_Vector     sgn = Q6_V_vor_VV(Q6_V_vsplat_R(0x3C00), Q6_Vw_vasr_VwR(Q6_V_vand_VV(wv, Q6_V_vsplat_R(0x80000000)), 16));
    const HVX_Vector     sv  = Q6_Vh_vpacke_VwVw(zero, Q6_V_vmux_QVV(nz, sgn, zero));

    // token column of each row; -1 for missing rows (only rows 0..31 are used)
    int16_t tcol[64] __attribute__((aligned(128))) = { 0 };
    for (uint32_t r = 0; r < 32; ++r) {
        tcol[r] = r < nr ? (int16_t) (t % 32) : -1;
        if (++h == nh) {
            h = 0;
            t++;
        }
    }
    const HVX_Vector tv = *(const HVX_Vector *) tcol;

    // halfword 2n and 2n + 1 of a G vector are column n
    static const int16_t col[64] __attribute__((aligned(128))) = {
        0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10,
        11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21,
        22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31,
    };
    const HVX_Vector cv = *(const HVX_Vector *) col;

    // vector p holds rows 2p, 2p + 1
    HVX_Vector * gv = (HVX_Vector *) gt;
    for (uint32_t p = 0; p < 16; ++p) {
        const HVX_Vector tp = hvx_vec_repl4(Q6_V_vror_VR(tv, p * 4));
        const HVX_Vector sp = hvx_vec_repl4(Q6_V_vror_VR(sv, p * 4));
        gv[p] = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(cv, tp), sp, zero);
    }
}

// Q buffer slot of set
static inline uint32_t lid_hmx_slot(const struct htp_lid_hmx_context * hctx, uint32_t qbuf, uint32_t set) {
    return qbuf * hctx->n_qsets + set;
}

// Scales and G of channel tile c of a Q slot (needs the slot's w)
static inline void lid_hmx_q_scg(const struct htp_lid_hmx_context * hctx, uint32_t slot, const struct lid_hmx_set * S, uint32_t c) {
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;
    const uint32_t nh = hctx->octx->src[0]->ne[1];
    const uint32_t ch = c * 32;
    const uint32_t t  = fastdiv(ch, &hctx->kp->div_n_head);

    lid_hmx_build_sc_g(hctx->sc + (slot * L->n_ch + c) * 256, hctx->gw + (slot * L->n_ch + c) * HMX_FP16_TILE_N_ELMS,
                       hctx->w + slot * (L->size_w / sizeof(float)) + ch, t, ch - t * nh, MIN(32, S->nt * nh - ch), nh);
}

// Staged f32 Q rows of channel tile c to f16, then to the weight tiles of a Q slot
static inline void lid_hmx_q_tiles(const struct htp_lid_hmx_context * hctx, uint32_t slot, const struct lid_hmx_set * S, uint32_t c,
                                   const uint8_t * qs, uint8_t * qh) {
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;
    const uint32_t D  = hctx->octx->src[0]->ne[0];
    const uint32_t nh = hctx->octx->src[0]->ne[1];
    const uint32_t nr = MIN(32, S->nt * nh - c * 32);

    for (uint32_t r = 0; r < nr; ++r) {
        hvx_copy_f16_f32_aa(qh + r * L->size_qh_row, qs + r * L->size_qs_row, D);
    }

    // missing channels must be zero (finite): zero the tiles, then scatter only the valid rows
    __fp16 * qw = hctx->qw + (slot * L->n_ch + c) * L->n_dot * HMX_FP16_TILE_N_ELMS;
    if (nr < 32) {
        for (uint32_t i = 0; i < L->n_dot * 16; ++i) {
            ((HVX_Vector *) qw)[i] = Q6_V_vzero();
        }
    }
    hmx_interleave_rows_to_tiles(qw, (const __fp16 *) qh, nr, D, L->size_qh_row / sizeof(__fp16), 0, hex_round_up(nr, 2));
    hvx_gather_sync(qw);
}

// Stage g share of the next run's Q prep (the DMA queue of the thread is empty here)
static void lid_hmx_q_chunk(const struct htp_lid_hmx_context * hctx, uint32_t ith, uint32_t nth) {
    const struct htp_ops_context * octx = hctx->octx;
    const struct htp_tensor *      w    = octx->src[2];
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;

    const uint32_t nh     = octx->src[0]->ne[1];
    const uint32_t n_tot  = hctx->nx_n_run * L->n_ch;
    const uint32_t a      = fastdiv(hctx->g * n_tot, &hctx->nx_stg_div);
    const uint32_t b      = fastdiv((hctx->g + 1) * n_tot, &hctx->nx_stg_div);
    const uint32_t per    = fastdiv(b - a + octx->n_threads - 1, &octx->n_threads_div);
    const uint32_t gc0    = MIN(a + ith * per, b);
    const uint32_t gc1    = MIN(gc0 + per, b);
    const uint32_t qb     = hctx->qbuf ^ 1;

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint8_t *   qs    = hctx->qs + ith * L->size_qs;
    uint8_t *   qh    = qs + 32 * L->size_qs_row;

    for (uint32_t gc = gc0; gc < gc1; ++gc) {
        const uint32_t             set = fastdiv(gc, &hctx->kp->div_n_ch);
        const uint32_t             c   = gc - set * L->n_ch;
        const struct lid_hmx_set * S   = &hctx->nx_sets[set];
        if (c >= S->n_ch) {
            continue;
        }

        const uint32_t slot = lid_hmx_slot(hctx, qb, set);
        const uint32_t ch   = c * 32;
        const uint32_t ta   = fastdiv(ch, &hctx->kp->div_n_head);
        const uint32_t tb   = fastdiv(MIN(ch + 32, S->nt * nh) - 1, &hctx->kp->div_n_head) + 1;

        float * wd = hctx->w + slot * (L->size_w / sizeof(float));
        dma_queue_push(dma_q, dma_make_data(wd + ta * nh, w->data + S->is * w->nb[3] + (S->t0 + ta) * w->nb[1]),
                       nh * sizeof(float), w->nb[1], nh * sizeof(float), tb - ta);
        const uint32_t n_dma = 1 + lid_hmx_push_q(hctx, dma_q, qs, S, c);
        for (uint32_t i = 0; i < n_dma; ++i) {
            dma_queue_pop(dma_q);
        }

        lid_hmx_q_scg(hctx, slot, S, c);
        lid_hmx_q_tiles(hctx, slot, S, c, qs, qh);
    }
}

// Q rows to weight tiles, per channel scales |w| and G tiles for all sets of the run, then stage -1 (prep block 0).
// Q work goes to the first threads (thread 0 starts first), the K rows of block 0 to the others.
static void lid_hmx_qprep_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_lid_hmx_context *   hctx = (struct htp_lid_hmx_context *) data;
    const struct htp_ops_context * octx = hctx->octx;
    const struct htp_tensor *      q    = octx->src[0];
    const struct htp_tensor *      w    = octx->src[2];
    const struct htp_lid_hmx_vtcm_layout * L = &hctx->L;

    const uint32_t nh = q->ne[1];

    struct htp_thread_trace * tr    = &octx->ctx->trace[ith];
    dma_queue *               dma_q = octx->ctx->dma[ith];

    const uint32_t n_tot = hctx->n_run * L->n_ch;
    const uint32_t per   = fastdiv(n_tot + octx->n_threads - 1, &octx->n_threads_div);
    const uint32_t gc0   = MIN(ith * per, n_tot);
    const uint32_t gc1   = MIN(gc0 + per, n_tot);

    uint8_t * qs = hctx->qs + ith * L->size_qs;
    uint8_t * qh = qs + 32 * L->size_qs_row;

    // all input DMAs in flight at once: w of the sets, Q of the first channel tile, then K and mask of block 0
    uint32_t n_dma = 0;
    uint32_t first = gc1;
    const struct fastdiv_values * div_n_ch = &hctx->kp->div_n_ch;
    for (uint32_t set = fastdiv(gc0, div_n_ch); gc0 < gc1 && set <= fastdiv(gc1 - 1, div_n_ch); ++set) {
        const struct lid_hmx_set S = hctx->sets[set];

        const uint32_t c0 = MAX(gc0, set * L->n_ch) - set * L->n_ch;
        const uint32_t c1 = MIN(MIN(gc1, (set + 1) * L->n_ch) - set * L->n_ch, S.n_ch);
        if (c0 >= c1) {
            continue;
        }
        first = MIN(first, set * L->n_ch + c0);

        const uint32_t ta = fastdiv(c0 * 32, &hctx->kp->div_n_head);
        const uint32_t tb = fastdiv(MIN(c1 * 32, S.nt * nh) - 1, &hctx->kp->div_n_head) + 1;
        float * wd = hctx->w + lid_hmx_slot(hctx, hctx->qbuf, set) * (L->size_w / sizeof(float));
        dma_queue_push(dma_q, dma_make_data(wd + ta * nh, w->data + S.is * w->nb[3] + (S.t0 + ta) * w->nb[1]),
                       nh * sizeof(float), w->nb[1], nh * sizeof(float), tb - ta);
        n_dma++;
    }

    if (first < gc1) {
        const uint32_t           set = fastdiv(first, div_n_ch);
        const struct lid_hmx_set S   = hctx->sets[set];
        n_dma += lid_hmx_push_q(hctx, dma_q, qs, &S, first - set * L->n_ch);
    }

    uint32_t n_in = lid_hmx_push_in(hctx, ith, nth, 0);

    for (uint32_t i = 0; i < n_dma; ++i) {
        dma_queue_pop(dma_q);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) gc0);

    for (uint32_t gc = first; gc < gc1; ++gc) {
        const uint32_t           set = fastdiv(gc, div_n_ch);
        const uint32_t           c   = gc - set * L->n_ch;
        const struct lid_hmx_set S   = hctx->sets[set];
        if (c >= S.n_ch) {
            continue;
        }

        uint32_t n_q = 0;
        if (gc != first) {
            n_q  = lid_hmx_push_q(hctx, dma_q, qs, &S, c) + n_in;
            n_in = 0;
        }

        // scales and G while the Q rows land
        const uint32_t slot = lid_hmx_slot(hctx, hctx->qbuf, set);
        lid_hmx_q_scg(hctx, slot, &S, c);

        for (uint32_t i = 0; i < n_q; ++i) {
            dma_queue_pop(dma_q);
        }

        lid_hmx_q_tiles(hctx, slot, &S, c, qs, qh);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) gc0);

    for (uint32_t i = 0; i < n_in; ++i) {
        dma_queue_pop(dma_q);
    }

    lid_hmx_stage(hctx, ith, nth);
}

// Run start when its Q was prepped in the previous run: block 0 K and mask, then stage -1
static void lid_hmx_first_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_lid_hmx_context * hctx = (const struct htp_lid_hmx_context *) data;

    const uint32_t n_in = lid_hmx_push_in(hctx, ith, nth, 0);
    for (uint32_t i = 0; i < n_in; ++i) {
        dma_queue_pop(hctx->octx->ctx->dma[ith]);
    }
    lid_hmx_stage(hctx, ith, nth);
}

static int lid_hmx_run(struct htp_ops_context * octx, const struct htp_lid_kernel_params * kparams) {
    const struct htp_tensor * q = octx->src[0];

    const uint32_t D         = q->ne[0];
    const uint32_t n_head    = q->ne[1];
    const uint32_t n_stream  = q->ne[3];
    const uint32_t n_threads = kparams->n_threads;
    const uint32_t n_qsets   = kparams->n_qsets;

    struct htp_lid_hmx_context hctx;
    memset(&hctx, 0, sizeof(hctx));
    hctx.octx    = octx;
    hctx.kp      = kparams;
    hctx.Bc      = kparams->kv_block;
    hctx.kv_seg  = kparams->kv_seg;
    hctx.n_tile  = kparams->n_tokens_tile;
    hctx.n_tt    = kparams->n_tt;
    hctx.n_qsets = n_qsets;
    htp_lid_hmx_vtcm_layout_build(&hctx.L, D, n_head, hctx.n_tile, hctx.Bc, n_qsets * kparams->n_qbuf, n_threads,
                                  kparams->k_row_size);

    uint8_t * const base = octx->ctx->vtcm_base;
    const struct htp_lid_hmx_vtcm_layout * L = &hctx.L;

    hctx.qw    = VTCM_LAYOUT_PTR(__fp16, base, L->off_qw);
    hctx.gw    = VTCM_LAYOUT_PTR(__fp16, base, L->off_g);
    hctx.sc    = VTCM_LAYOUT_PTR(uint8_t, base, L->off_sc);
    hctx.sc_id = VTCM_LAYOUT_PTR(uint8_t, base, L->off_sc_id);
    hctx.s     = VTCM_LAYOUT_PTR(__fp16, base, L->off_s);
    hctx.w     = VTCM_LAYOUT_PTR(float, base, L->off_w);
    hctx.qs    = VTCM_LAYOUT_PTR(uint8_t, base, L->off_qs);
    for (int i = 0; i < 2; i++) {
        hctx.k_raw[i] = VTCM_LAYOUT_PTR(uint8_t, base, L->off_k_raw[i]);
        hctx.k_t[i]   = VTCM_LAYOUT_PTR(__fp16, base, L->off_k_t[i]);
        hctx.o[i]     = VTCM_LAYOUT_PTR(__fp16, base, L->off_o[i]);
        hctx.out[i]   = VTCM_LAYOUT_PTR(uint8_t, base, L->off_out[i]);
    }
    for (int i = 0; i < 4; i++) {
        hctx.m[i] = VTCM_LAYOUT_PTR(uint8_t, base, L->off_m[i]);
    }

    hmx_init_column_scales(hctx.sc_id, Q6_V_vsplat_R(0x3c00)); // identity 1.0 f16

    hmx_queue_t   hmx_q = octx->ctx->hmx_queue;
    lid_hmx_job_t job[2];

    const uint32_t n_tiles = n_stream * hctx.n_tt;

    for (uint32_t tile0 = 0, r = 0; tile0 < n_tiles; tile0 += n_qsets, ++r) {
        hctx.tile0 = tile0;
        hctx.n_run = MIN(n_qsets, n_tiles - tile0);
        hctx.qbuf  = r & (kparams->n_qbuf - 1); // n_qbuf is 1 or 2

        for (uint32_t i = 0; i < hctx.n_run; ++i) {
            hctx.sets[i] = lid_hmx_set_init(&hctx, i);
        }

        // all sets of a run have the same token count: one set per run, or all tokens in one tile
        const struct lid_hmx_set S0 = hctx.sets[0];
        hctx.nt_run   = S0.nt;
        hctx.n_u_run  = S0.n_u;
        hctx.n_blocks = hctx.n_run == n_qsets ? kparams->n_kv_blocks : (hctx.n_run * hctx.kv_seg + hctx.Bc - 1) / hctx.Bc;

        const int32_t  n     = (int32_t) hctx.n_blocks;
        const uint32_t n_tot = hctx.n_run * L->n_ch;
        const uint32_t per   = fastdiv(n_tot + n_threads - 1, &octx->n_threads_div);

        hctx.n_qthr = (n_tot + per - 1) / per;
        hctx.g      = -1;

        const bool q_ready = hctx.nx_on;
        if (q_ready) {
            hctx.n_qthr = 0;
            hctx.nx_on  = false;
            work_queue_run(octx->ctx->work_queue, lid_hmx_first_thread, &hctx, n_threads);
        } else {
            work_queue_run(octx->ctx->work_queue, lid_hmx_qprep_thread, &hctx, n_threads);
        }

        // with two Q buffers the stages of this run prep the Q of the next one
        if (kparams->n_qbuf == 2 && tile0 + n_qsets < n_tiles) {
            struct htp_lid_hmx_context nx = hctx;
            nx.tile0      = tile0 + n_qsets;
            hctx.nx_n_run = MIN(n_qsets, n_tiles - nx.tile0);
            for (uint32_t i = 0; i < hctx.nx_n_run; ++i) {
                hctx.nx_sets[i] = lid_hmx_set_init(&nx, i);
            }
            hctx.nx_stg_div = init_fastdiv_values(hctx.n_blocks + 1);
            hctx.nx_on      = true;
        }

        for (int32_t g = 0; g < n; ++g) {
            struct lid_hmx_seg seg[LID_HMX_MAX_SEGS];
            uint32_t           n_rows;
            const uint32_t     n_seg = lid_hmx_segs(&hctx, g, seg, &n_rows);

            lid_hmx_job_t * j = &job[g % 2];
            #pragma clang loop vectorize(disable)
            for (uint32_t i = 0; i < n_seg; ++i) {
                const uint32_t slot = lid_hmx_slot(&hctx, hctx.qbuf, seg[i].set);
                j->seg[i].qw   = hctx.qw + slot * L->n_ch * L->n_dot * HMX_FP16_TILE_N_ELMS;
                j->seg[i].gw   = hctx.gw + slot * L->n_ch * HMX_FP16_TILE_N_ELMS;
                j->seg[i].sc   = hctx.sc + slot * L->n_ch * 256;
                j->seg[i].rt0  = seg[i].x0 / 32;
                j->seg[i].n_rt = (seg[i].nv + 31) / 32;
            }
            j->n_seg  = n_seg;
            j->k_t    = hctx.k_t[g % 2];
            j->sc_id  = hctx.sc_id;
            j->s      = hctx.s;
            j->o      = hctx.o[g % 2];
            j->n_ch   = S0.n_ch;
            j->n_u    = S0.n_u;
            j->n_dot  = L->n_dot;
            j->n_head = n_head;
            j->n_sg   = L->n_sg;
            hmx_queue_push(hmx_q, hmx_queue_make_desc(lid_hmx_worker, j));

            // stage g has work only if there is a next block, a previous one or next run Q work
            hctx.g = g;
            if (g + 1 < n || g >= 1 || hctx.nx_on) {
                work_queue_run(octx->ctx->work_queue, lid_hmx_stage_thread, &hctx, n_threads);
            }

            hmx_queue_pop(hmx_q);
        }

        // last stage only extracts; with one token only thread 0 has work (same token split for nth 1)
        hctx.g = n;
        const uint32_t n_last = hctx.nt_run == 1 && !hctx.nx_on ? 1 : n_threads;
        work_queue_run(octx->ctx->work_queue, lid_hmx_stage_thread, &hctx, n_last);
    }

    return HTP_STATUS_OK;
}

int op_lightning_indexer(struct htp_ops_context * octx) {
    const struct htp_tensor * q = octx->src[0];
    const struct htp_tensor * m = octx->src[3];

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

    if (kparams->kernel_type == HTP_LID_KERNEL_HMX) {
        return lid_hmx_run(octx, kparams);
    }

    struct htp_lid_context lctx;
    lctx.octx              = octx;
    lctx.kp                = kparams;
    lctx.n_blocks          = kparams->n_kv_blocks;
    lctx.blocks_per_thread = kparams->blocks_per_thread;

    htp_lid_vtcm_layout_build(&lctx.L, q->ne[0], q->ne[1], kparams->n_tokens_tile, kparams->n_threads, kparams->k_row_size);

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
        lctx.im3 = fastmodulo(is, m->ne[3], &kparams->div_m_ne3);

        for (uint32_t t0 = 0; t0 < n_tokens; t0 += n_tile) {
            lctx.t0 = t0;
            lctx.nt = MIN(n_tile, n_tokens - t0);

            work_queue_run(octx->ctx->work_queue, lid_prep_thread, &lctx, MIN(n_threads, lctx.nt * q->ne[1]));
            work_queue_run(octx->ctx->work_queue, lid_compute_thread, &lctx, n_threads);
        }
    }

    return HTP_STATUS_OK;
}
