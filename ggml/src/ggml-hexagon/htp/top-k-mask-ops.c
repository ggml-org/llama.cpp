#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>

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
#include "top-k-mask-ops.h"

#define TOP_K_MASK_NINF_F32 0xff800000u // f32 score compare

struct htp_top_k_mask_context {
    struct htp_ops_context *                    octx;
    const struct htp_top_k_mask_kernel_params * kparams;

    uint32_t rows_per_hvx;
    uint32_t row0, row1;           // rows this device handles
    uint32_t off_extra, off_score; // parts inside one row input slot
    uint32_t col0, ncols_out;      // PICKS: columns this device writes back
    atomic_uint barrier;           // PICKS: arrivals, barrier k waits for k * n_hvx

    uint8_t * blocks;
    uint8_t * mask;
    uint8_t * rowin;
    uint8_t * tiles;
    uint8_t * mtiles;
    uint8_t * scratch;
};

// DMA queue pop with a trace event around the wait
static inline void top_k_mask_pop(dma_queue * dma_q, struct htp_thread_trace * tr, uint16_t info) {
    htp_trace_event_start(tr, HTP_TRACE_EVT_DMA, info);
    dma_queue_pop(dma_q);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_DMA, info);
}

// Lanes 0..31 as words, in registers: inclusive scan of ones, minus one
static inline HVX_Vector top_k_mask_lane_iota(void) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    HVX_Vector       v    = one;
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 4));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vlalign_VVR(v, zero, 64));
    return Q6_Vw_vsub_VwVw(v, one);
}

static __attribute__((noinline)) void top_k_mask_fill(uint8_t * restrict dst, uint32_t nbytes, uint32_t value) {
    const HVX_Vector v    = Q6_V_vsplat_R(value);
    HVX_Vector *     vdst = (HVX_Vector *) dst;
    const uint32_t   nvec = nbytes / VLEN;

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        vdst[i] = v;
    }
}

#define TOP_K_MASK_DEAD 0x7fffffffu // cell value of a dead lane: c - c0 >=u ncols for every tile, signed max

// Byte offsets of the cells in [c0, c0 + ncols) relative to c0, 0 for the others
static inline HVX_Vector top_k_mask_keep_off(HVX_Vector c, HVX_Vector v_c0, HVX_Vector v_ncols, const bool is_f16) {
    const HVX_Vector cr = Q6_Vw_vsub_VwVw(c, v_c0);
    return Q6_Vw_vasl_VwR(Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(v_ncols, cr), cr, Q6_V_vzero()), is_f16 ? 1 : 2);
}

// Mask value of each cell lane in [c0, c0 + ncols) into tmp (predicated gather), returns the scatter offsets
static inline HVX_Vector top_k_mask_keep_gather(HVX_Vector c, const uint8_t * restrict mrow, HVX_Vector v_c0, HVX_Vector v_ncols,
                                                uint32_t mu, HVX_Vector * restrict tmp, const bool is_f16) {
    const HVX_VectorPred q   = Q6_Q_vcmp_gt_VuwVuw(v_ncols, Q6_Vw_vsub_VwVw(c, v_c0));
    const HVX_Vector     off = top_k_mask_keep_off(c, v_c0, v_ncols, is_f16);

    // Halfword data with word offsets: both halves of the pair hold the same offsets, the word predicate covers both halfwords of a lane
    if (is_f16) {
        Q6_vgather_AQRMWw(tmp, q, (size_t) mrow, mu, Q6_W_vcombine_VV(off, off));
    } else {
        Q6_vgather_AQRMVw(tmp, q, (size_t) mrow, mu, off);
    }
    return off;
}

static inline void top_k_mask_keep_scatter(HVX_Vector c, HVX_Vector off, uint8_t * restrict tile, HVX_Vector v_c0, HVX_Vector v_ncols,
                                           uint32_t mu, HVX_Vector val, const bool is_f16) {
    const HVX_VectorPred q = Q6_Q_vcmp_gt_VuwVuw(v_ncols, Q6_Vw_vsub_VwVw(c, v_c0));
    if (is_f16) {
        Q6_vscatter_QRMWwV(q, (size_t) tile, mu, Q6_W_vcombine_VV(off, off), val);
    } else {
        Q6_vscatter_QRMVwV(q, (size_t) tile, mu, off, val);
    }
}

// Keep 4 cell vectors (dead lanes = TOP_K_MASK_DEAD): tile[c - c0] = mrow[c - c0] for c in [c0, c0 + ncols). One gather sync.
// flag: no mask, tile[c - c0] = all ones (keep flag)
static inline void top_k_mask_keep4(HVX_Vector c0v, HVX_Vector c1v, HVX_Vector c2v, HVX_Vector c3v, uint8_t * restrict tile,
                                    const uint8_t * restrict mrow, HVX_Vector v_c0, HVX_Vector v_ncols, uint32_t mu,
                                    HVX_Vector * restrict tmp, const bool is_f16, const bool flag) {
    if (flag) {
        const HVX_Vector ones = Q6_V_vsplat_R(0xffffffffu);
        top_k_mask_keep_scatter(c0v, top_k_mask_keep_off(c0v, v_c0, v_ncols, is_f16), tile, v_c0, v_ncols, mu, ones, is_f16);
        top_k_mask_keep_scatter(c1v, top_k_mask_keep_off(c1v, v_c0, v_ncols, is_f16), tile, v_c0, v_ncols, mu, ones, is_f16);
        top_k_mask_keep_scatter(c2v, top_k_mask_keep_off(c2v, v_c0, v_ncols, is_f16), tile, v_c0, v_ncols, mu, ones, is_f16);
        top_k_mask_keep_scatter(c3v, top_k_mask_keep_off(c3v, v_c0, v_ncols, is_f16), tile, v_c0, v_ncols, mu, ones, is_f16);
        return;
    }
    const HVX_Vector o0 = top_k_mask_keep_gather(c0v, mrow, v_c0, v_ncols, mu, tmp + 0, is_f16);
    const HVX_Vector o1 = top_k_mask_keep_gather(c1v, mrow, v_c0, v_ncols, mu, tmp + 1, is_f16);
    const HVX_Vector o2 = top_k_mask_keep_gather(c2v, mrow, v_c0, v_ncols, mu, tmp + 2, is_f16);
    const HVX_Vector o3 = top_k_mask_keep_gather(c3v, mrow, v_c0, v_ncols, mu, tmp + 3, is_f16);
    hvx_gather_sync(tmp + 3);
    top_k_mask_keep_scatter(c0v, o0, tile, v_c0, v_ncols, mu, tmp[0], is_f16);
    top_k_mask_keep_scatter(c1v, o1, tile, v_c0, v_ncols, mu, tmp[1], is_f16);
    top_k_mask_keep_scatter(c2v, o2, tile, v_c0, v_ncols, mu, tmp[2], is_f16);
    top_k_mask_keep_scatter(c3v, o3, tile, v_c0, v_ncols, mu, tmp[3], is_f16);
}

// Pick vector j: picks below j1 and n_pick are valid, returns them with dead lanes marked. v_rem = j1 - lane index.
static inline HVX_Vector top_k_mask_load_picks(const int32_t * restrict tk, uint32_t j, HVX_Vector v_rem, HVX_Vector v_npick) {
    const HVX_Vector     p_raw = hvx_vmem(tk + j);
    // lane + j <u j1 is the tail, p <u n_pick drops p < 0 and p >= n_pick
    const HVX_VectorPred valid = Q6_Q_vcmp_gtand_QVuwVuw(Q6_Q_vcmp_gt_VwVw(v_rem, Q6_V_vsplat_R(j)), v_npick, p_raw);
    return Q6_V_vmux_QVV(valid, p_raw, Q6_V_vsplat_R(TOP_K_MASK_DEAD));
}

// Scatter the kept cells of picks [j0, j1) (and the extras) into the tile of columns [c0, c0 + ncols). slot holds the row inputs, mrow the mask columns.
// Gathers are issued in batches with one sync per batch: tmp[0..3] scores/blocks, tmp[4] score with blocks, tmp[5..8] mask values.
static inline __attribute__((always_inline)) void top_k_mask_scatter_tile(
        const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot, const uint8_t * restrict mrow, uint8_t * restrict tile,
        HVX_Vector * restrict tmp, uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra,
        const bool is_f16, const bool flag) {
    const struct htp_top_k_mask_kernel_params * kp = tctx->kparams;

    const int32_t * restrict tk  = (const int32_t *) slot;
    const int32_t * restrict ex  = (const int32_t *) (slot + tctx->off_extra);
    const uint8_t * restrict sc  = slot + tctx->off_score;
    const uint8_t * restrict blk = tctx->blocks;

    const bool     has_score  = kp->flags & HTP_TOP_K_MASK_HAS_SCORE;
    const bool     has_blocks = kp->flags & HTP_TOP_K_MASK_HAS_BLOCKS;
    const uint32_t n_blk      = kp->n_blk;
    const uint32_t n_extra    = do_extra ? kp->n_extra : 0;
    const uint32_t mu         = ncols * kp->elem_size - 1;
    const uint32_t sc_mu      = kp->score_bytes - 1;
    const uint32_t blk_mu     = kp->blocks_bytes - 1;
    const uint32_t nb1_h      = ((uint32_t) kp->blocks_nb1 << 16) | kp->blocks_nb1;

    const HVX_Vector v_lane  = top_k_mask_lane_iota(); // in the HVX_COMP scope of the caller
    const HVX_Vector v_c0    = Q6_V_vsplat_R(c0);
    const HVX_Vector v_ncols = Q6_V_vsplat_R(ncols);
    const HVX_Vector v_npick = Q6_V_vsplat_R(kp->n_pick);
    const HVX_Vector v_pmax  = Q6_V_vsplat_R(kp->n_pick - 1); // dead picks gather the last entry, every gather stays in its region
    const HVX_Vector v_rem   = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(j1), v_lane);
    const HVX_Vector v_ninf  = Q6_V_vsplat_R(TOP_K_MASK_NINF_F32);
    const HVX_Vector v_dead  = Q6_V_vsplat_R(TOP_K_MASK_DEAD);

    HVX_Vector * restrict mtmp = tmp + 5;

    if (has_blocks) {
        for (uint32_t j = j0; j < j1; j += VLEN_FP32) {
            HVX_Vector       cell = top_k_mask_load_picks(tk, j, v_rem, v_npick);
            const HVX_Vector p    = Q6_Vw_vmin_VwVw(cell, v_pmax);
            if (has_score) {
                Q6_vgather_ARMVw(tmp + 4, (size_t) sc, sc_mu, Q6_Vw_vasl_VwR(p, 2));
            }
            const HVX_Vector boff = Q6_Vw_vmpyi_VwRh(p, nb1_h);
            for (uint32_t k = 0; k < n_blk; k += 4) {
                // blocks k..k+3 of every pick; the missing ones (k + i >= n_blk) are dead
                const uint32_t   nk = MIN(n_blk - k, 4);
                const HVX_Vector bo = Q6_Vw_vadd_VwVw(boff, Q6_V_vsplat_R(k * 4));
                for (uint32_t i = 0; i < nk; i++) {
                    Q6_vgather_ARMVw(tmp + i, (size_t) blk, blk_mu, Q6_Vw_vadd_VwVw(bo, Q6_V_vsplat_R(i * 4)));
                }
                hvx_gather_sync(tmp + nk - 1);
                if (has_score && k == 0) {
                    // IEEE compare: NaN and -inf scores are dead
                    cell = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tmp[4], v_ninf), cell, v_dead);
                }
                const HVX_VectorPred dead = Q6_Q_vcmp_eq_VwVw(cell, v_dead);
                top_k_mask_keep4(Q6_V_vmux_QVV(dead, v_dead, tmp[0]),
                                 nk > 1 ? Q6_V_vmux_QVV(dead, v_dead, tmp[1]) : v_dead,
                                 nk > 2 ? Q6_V_vmux_QVV(dead, v_dead, tmp[2]) : v_dead,
                                 nk > 3 ? Q6_V_vmux_QVV(dead, v_dead, tmp[3]) : v_dead,
                                 tile, mrow, v_c0, v_ncols, mu, mtmp, is_f16, flag);
            }
        }
    } else {
        // 4 pick vectors per batch, the vectors at or past j1 are dead and not loaded
        for (uint32_t j = j0; j < j1; j += 4 * VLEN_FP32) {
            HVX_Vector c0v = top_k_mask_load_picks(tk, j, v_rem, v_npick);
            HVX_Vector c1v = j + 1 * VLEN_FP32 < j1 ? top_k_mask_load_picks(tk, j + 1 * VLEN_FP32, v_rem, v_npick) : v_dead;
            HVX_Vector c2v = j + 2 * VLEN_FP32 < j1 ? top_k_mask_load_picks(tk, j + 2 * VLEN_FP32, v_rem, v_npick) : v_dead;
            HVX_Vector c3v = j + 3 * VLEN_FP32 < j1 ? top_k_mask_load_picks(tk, j + 3 * VLEN_FP32, v_rem, v_npick) : v_dead;
            if (has_score) {
                Q6_vgather_ARMVw(tmp + 0, (size_t) sc, sc_mu, Q6_Vw_vasl_VwR(Q6_Vw_vmin_VwVw(c0v, v_pmax), 2));
                Q6_vgather_ARMVw(tmp + 1, (size_t) sc, sc_mu, Q6_Vw_vasl_VwR(Q6_Vw_vmin_VwVw(c1v, v_pmax), 2));
                Q6_vgather_ARMVw(tmp + 2, (size_t) sc, sc_mu, Q6_Vw_vasl_VwR(Q6_Vw_vmin_VwVw(c2v, v_pmax), 2));
                Q6_vgather_ARMVw(tmp + 3, (size_t) sc, sc_mu, Q6_Vw_vasl_VwR(Q6_Vw_vmin_VwVw(c3v, v_pmax), 2));
                hvx_gather_sync(tmp + 3);
                c0v = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tmp[0], v_ninf), c0v, v_dead);
                c1v = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tmp[1], v_ninf), c1v, v_dead);
                c2v = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tmp[2], v_ninf), c2v, v_dead);
                c3v = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tmp[3], v_ninf), c3v, v_dead);
            }
            top_k_mask_keep4(c0v, c1v, c2v, c3v, tile, mrow, v_c0, v_ncols, mu, mtmp, is_f16, flag);
        }
    }

    const HVX_Vector v_rem_ex = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(n_extra), v_lane);
    for (uint32_t j = 0; j < n_extra; j += VLEN_FP32) {
        const HVX_VectorPred tail = Q6_Q_vcmp_gt_VwVw(v_rem_ex, Q6_V_vsplat_R(j));
        top_k_mask_keep4(Q6_V_vmux_QVV(tail, hvx_vmem(ex + j), v_dead), v_dead, v_dead, v_dead, tile, mrow, v_c0, v_ncols, mu, mtmp, is_f16, flag);
    }

    // release the scatters before DMA reads the tile
    hvx_gather_sync(tile);
}

static __attribute__((noinline)) void top_k_mask_scatter_tile_f32(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        const uint8_t * restrict mrow, uint8_t * restrict tile, HVX_Vector * restrict tmp,
                                        uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra) {
    top_k_mask_scatter_tile(tctx, slot, mrow, tile, tmp, c0, ncols, j0, j1, do_extra, false, false);
}

static __attribute__((noinline)) void top_k_mask_scatter_tile_f16(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        const uint8_t * restrict mrow, uint8_t * restrict tile, HVX_Vector * restrict tmp,
                                        uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra) {
    top_k_mask_scatter_tile(tctx, slot, mrow, tile, tmp, c0, ncols, j0, j1, do_extra, true, false);
}

// PICKS: keep flags (all ones) of the picks [j0, j1) into the flag tile of the whole row, the element size picks the flag width
static __attribute__((noinline)) void top_k_mask_flag_tile_f32(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        uint8_t * restrict flags, HVX_Vector * restrict tmp, uint32_t j0, uint32_t j1, bool do_extra) {
    top_k_mask_scatter_tile(tctx, slot, NULL, flags, tmp, 0, tctx->kparams->n_kv, j0, j1, do_extra, false, true);
}

static __attribute__((noinline)) void top_k_mask_flag_tile_f16(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        uint8_t * restrict flags, HVX_Vector * restrict tmp, uint32_t j0, uint32_t j1, bool do_extra) {
    top_k_mask_scatter_tile(tctx, slot, NULL, flags, tmp, 0, tctx->kparams->n_kv, j0, j1, do_extra, true, true);
}

// out = flag ? out : -inf over nbytes (a multiple of 128) of the row tile
static __attribute__((noinline)) void top_k_mask_select(uint8_t * restrict out, const uint8_t * restrict flags, uint32_t nbytes, uint32_t ninf) {
    const HVX_Vector     v_ninf = Q6_V_vsplat_R(ninf);
    HVX_Vector *         vout   = (HVX_Vector *) out;
    const HVX_Vector *   vflag  = (const HVX_Vector *) flags;
    const uint32_t       nvec   = nbytes / VLEN;

    #pragma unroll(4)
    for (uint32_t i = 0; i < nvec; i++) {
        vout[i] = Q6_V_vxor_VV(v_ninf, Q6_V_vand_VV(Q6_V_vxor_VV(vout[i], v_ninf), vflag[i]));
    }
}

// Push the row inputs of flat row r into slot: top_k, extra, score. Returns the push count.
static inline uint32_t top_k_mask_push_rowin(const struct htp_top_k_mask_context * tctx, dma_queue * dma_q, uint32_t r, uint8_t * slot) {
    const struct htp_ops_context *              octx  = tctx->octx;
    const struct htp_top_k_mask_kernel_params * kp    = tctx->kparams;
    const struct htp_tensor *                   top_k = octx->src[1];
    const struct htp_tensor *                   score = octx->src[2];
    const struct htp_tensor *                   extra = octx->src[4];

    const uint32_t tk_bytes = kp->n_top * sizeof(int32_t);
    dma_queue_push(dma_q, dma_make_data(slot, top_k->data + r * kp->u.rows.rs_top), tk_bytes, tk_bytes, tk_bytes, 1);
    uint32_t n = 1;

    if (kp->flags & HTP_TOP_K_MASK_HAS_EXTRA) {
        const uint32_t ex_bytes = kp->n_extra * sizeof(int32_t);
        dma_queue_push(dma_q, dma_make_data(slot + tctx->off_extra, extra->data + r * kp->u.rows.rs_extra), ex_bytes, ex_bytes, ex_bytes, 1);
        n++;
    }

    if (kp->flags & HTP_TOP_K_MASK_HAS_SCORE) {
        dma_queue_push(dma_q, dma_make_data(slot + tctx->off_score, score->data + r * kp->u.rows.rs_score), kp->score_bytes, kp->score_bytes, kp->score_bytes, 1);
        n++;
    }

    return n;
}

static inline struct fastdiv_values top_k_mask_fd(const struct htp_top_k_mask_kernel_params * kp, uint32_t k) {
    const struct fastdiv_values fd = { kp->u.rows.m.dec.mp[k], kp->u.rows.m.dec.l[k] };
    return fd;
}

// Mask row of flat row r: (i1, i2, i3) of dst, each taken modulo the mask dims, rows dense above the mask ne1
static inline uint32_t top_k_mask_mask_row(const struct htp_top_k_mask_kernel_params * kp, uint32_t r) {
    const uint16_t *            ne  = kp->u.rows.m.dec.ne;
    const struct fastdiv_values fd1 = top_k_mask_fd(kp, 0);
    const struct fastdiv_values fd2 = top_k_mask_fd(kp, 1);
    const struct fastdiv_values fm1 = top_k_mask_fd(kp, 2);
    const struct fastdiv_values fm2 = top_k_mask_fd(kp, 3);
    const struct fastdiv_values fm3 = top_k_mask_fd(kp, 4);

    const uint32_t q1 = fastdiv(r, &fd1);
    const uint32_t i1 = r - q1 * ne[0];
    const uint32_t i3 = fastdiv(q1, &fd2);
    const uint32_t i2 = q1 - i3 * ne[1];

    const uint32_t m1 = fastmodulo(i1, ne[2], &fm1);
    const uint32_t m2 = fastmodulo(i2, ne[3], &fm2);
    const uint32_t m3 = fastmodulo(i3, ne[4], &fm3);
    return (m3 * ne[3] + m2) * ne[2] + m1;
}

// Push the inputs of item (row r, col tile ct): row inputs on the first tile of a row, then the mask tile. Returns the push count.
static inline uint32_t top_k_mask_push_item(const struct htp_top_k_mask_context * tctx, dma_queue * dma_q, uint32_t r, uint32_t ct,
                                            uint8_t * slot, uint8_t * mslot) {
    const struct htp_ops_context *              octx = tctx->octx;
    const struct htp_top_k_mask_kernel_params * kp   = tctx->kparams;
    const struct htp_tensor *                   mask = octx->src[0];

    const uint32_t c0 = ct * kp->tile_cols;

    uint32_t n = 0;
    if (ct == 0) {
        n += top_k_mask_push_rowin(tctx, dma_q, r, slot);
    }

    if (mslot) {
        const uint32_t m_off = top_k_mask_mask_row(kp, r) * kp->u.rows.mask_nb1 + c0 * kp->elem_size;
        const uint32_t nb    = MIN(kp->tile_cols, kp->n_kv - c0) * kp->elem_size;
        dma_queue_push(dma_q, dma_make_data(mslot, mask->data + m_off), nb, nb, nb, 1);
        n++;
    }

    return n;
}

static inline void top_k_mask_next(uint32_t * r, uint32_t * ct, uint32_t n_col_tiles) {
    if (++(*ct) == n_col_tiles) {
        *ct = 0;
        ++(*r);
    }
}

// ROWS: the thread count follows the device rows (mdev), so the slice of an n byte copy is sized here
static inline uint32_t top_k_mask_slice(const struct htp_ops_context * octx, unsigned int nth, uint32_t n) {
    return hex_round_up(fastdiv(n + nth - 1, &octx->n_hvx_div), HTP_TOP_K_MASK_UNIT);
}

// Push this thread's slice (128B aligned) of a contiguous n byte copy. Returns the push count.
static inline uint32_t top_k_mask_push_slice(dma_queue * dma_q, unsigned int ith, dma_addr_t dst, dma_addr_t src, uint32_t n, uint32_t slice) {
    const uint32_t b0    = ith * slice;
    if (b0 >= n) {
        return 0;
    }
    const uint32_t nb = MIN(slice, n - b0);
    dma_queue_push(dma_q, dma_make_data(dst + b0, src + b0), nb, nb, nb, 1);
    return 1;
}

// Pass 1 (ROWS): load the blocks table and the resident mask row, split across threads. Fill the tiles of this thread with -inf meanwhile.
static void top_k_mask_preload(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_top_k_mask_context *             tctx  = (struct htp_top_k_mask_context *) data;
    const struct htp_top_k_mask_kernel_params * kp    = tctx->kparams;
    struct htp_ops_context *                    octx  = tctx->octx;
    dma_queue *                                 dma_q = th->dma_q;
    struct htp_thread_trace *                   tr    = th->trace;

    uint32_t n_pushed = 0;

    if (kp->blocks_bytes) {
        n_pushed += top_k_mask_push_slice(dma_q, ith, (dma_addr_t) tctx->blocks, octx->src[3]->data, kp->blocks_bytes, top_k_mask_slice(octx, nth, kp->blocks_bytes));
    }
    if (kp->flags & HTP_TOP_K_MASK_MASK_RESIDENT) {
        const uint32_t mask_bytes = kp->u.rows.m.mask_bytes;
        n_pushed += top_k_mask_push_slice(dma_q, ith, (dma_addr_t) tctx->mask, octx->src[0]->data, mask_bytes, top_k_mask_slice(octx, nth, mask_bytes));
    }

    const uint32_t tile_bytes = kp->tile_units * HTP_TOP_K_MASK_UNIT;

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
    top_k_mask_fill(tctx->tiles + ith * 2 * tile_bytes, 2 * tile_bytes, kp->ninf);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t i = 0; i < n_pushed; i++) {
        top_k_mask_pop(dma_q, tr, (uint16_t) i);
    }
}

// Pass 2 (ROWS): depth-2 DMA pipeline over the items of rows [row_first, row_last), item = (row, col tile).
// The row inputs alternate between 2 slots by row.
static void top_k_mask_run_items(const struct htp_top_k_mask_context * tctx, const htp_thread_t * th, unsigned int ith, uint32_t row_first, uint32_t row_last) {
    struct htp_ops_context *                    octx  = tctx->octx;
    const struct htp_top_k_mask_kernel_params * kp    = tctx->kparams;
    const struct htp_tensor *                   dst   = octx->dst;
    dma_queue *                                 dma_q = th->dma_q;
    struct htp_thread_trace *                   tr    = th->trace;

    const bool     is_f16    = kp->flags & HTP_TOP_K_MASK_F16;
    const uint32_t elem_size = kp->elem_size;

    const uint32_t rowin_bytes = kp->rowin_units * HTP_TOP_K_MASK_UNIT;
    const uint32_t tile_bytes  = kp->tile_units * HTP_TOP_K_MASK_UNIT;
    const uint32_t tile_cols   = kp->tile_cols;
    const uint32_t n_col_tiles = kp->n_col_tiles;
    const uint32_t n_items     = (row_last - row_first) * n_col_tiles;

    // 2 row input slots per thread, alternating by row
    uint8_t * rowin  = tctx->rowin + ith * 2 * rowin_bytes;
    uint8_t * tiles  = tctx->tiles + ith * 2 * tile_bytes;
    uint8_t * mtiles = (kp->flags & HTP_TOP_K_MASK_MASK_RESIDENT) ? NULL : tctx->mtiles + ith * 2 * tile_bytes;

    HVX_Vector * tmp = (HVX_Vector *) (tctx->scratch + ith * HTP_TOP_K_MASK_SCRATCH_BYTES);

    const dma_addr_t dst_data = dst->data;

    uint32_t n_in[2] = { 0, 0 };

    // prefetch cursor
    uint32_t pr = row_first, pct = 0;

    for (uint32_t it = 0; it < MIN(n_items, 2); it++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_push(dma_q, dma_make_data(dst_data, tiles + it * tile_bytes), 0, 0, 0, 0);
        n_in[it] = top_k_mask_push_item(tctx, dma_q, pr, pct, rowin + ((pr - row_first) & 1) * rowin_bytes, mtiles ? mtiles + it * tile_bytes : NULL);
        top_k_mask_next(&pr, &pct, n_col_tiles);
    }

    uint32_t r = row_first, ct = 0;

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t s    = it & 1;
        uint8_t *      tile = tiles + s * tile_bytes;

        // writeback of item it - 2 (or the dummy), then the tile is free again
        top_k_mask_pop(dma_q, tr, (uint16_t) it);
        if (it >= 2) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);
            top_k_mask_fill(tile, tile_bytes, kp->ninf);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);
        }

        for (uint32_t i = 0; i < n_in[s]; i++) {
            top_k_mask_pop(dma_q, tr, (uint16_t) it);
        }

        const uint32_t c0    = ct * tile_cols;
        const uint32_t ncols = MIN(tile_cols, kp->n_kv - c0);

        const uint8_t * slot = rowin + ((r - row_first) & 1) * rowin_bytes;
        const uint8_t * mrow = mtiles ? mtiles + s * tile_bytes : tctx->mask + c0 * elem_size;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);
        if (is_f16) {
            top_k_mask_scatter_tile_f16(tctx, slot, mrow, tile, tmp, c0, ncols, 0, kp->n_top, true);
        } else {
            top_k_mask_scatter_tile_f32(tctx, slot, mrow, tile, tmp, c0, ncols, 0, kp->n_top, true);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);

        const uint32_t d_off = r * kp->u.rows.rs_dst + c0 * elem_size;
        const uint32_t nb    = ncols * elem_size;
        dma_queue_push(dma_q, dma_make_data(dst_data + d_off, tile), nb, nb, nb, 1);

        if (it + 2 < n_items) {
            n_in[s] = top_k_mask_push_item(tctx, dma_q, pr, pct, rowin + ((pr - row_first) & 1) * rowin_bytes, mtiles ? mtiles + s * tile_bytes : NULL);
            top_k_mask_next(&pr, &pct, n_col_tiles);
        }
        top_k_mask_next(&r, &ct, n_col_tiles);
    }

    const uint32_t n_wb = MIN(n_items, 2);
    for (uint32_t i = 0; i < n_wb; i++) {
        top_k_mask_pop(dma_q, tr, (uint16_t) i);
    }
}

// Pass 2 (ROWS): whole rows split across threads, the col tiles of a row in order
static void top_k_mask_thread_rows(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    const struct htp_top_k_mask_context * tctx = (const struct htp_top_k_mask_context *) data;

    const uint32_t ir0 = tctx->row0 + ith * tctx->rows_per_hvx;
    const uint32_t ir1 = MIN(ir0 + tctx->rows_per_hvx, tctx->row1);
    if (ir0 >= ir1) {
        return;
    }

    top_k_mask_run_items(tctx, th, ith, ir0, ir1);
}

static inline void top_k_mask_barrier(atomic_uint * b, uint32_t target, struct htp_thread_trace * tr, uint16_t info) {
    htp_trace_event_start(tr, HTP_TRACE_EVT_FENCE, info);
    atomic_fetch_add_explicit(b, 1, memory_order_release);
    while (atomic_load_explicit(b, memory_order_acquire) < target) {
        hex_pause();
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_FENCE, info);
}

// PICKS, one pass over n_virt = kp->n_hvx virtual workers (host slices); real worker ith runs the virtual workers ith, ith + nth, ...
// A: shared preload and a slice of the flag clear per virtual worker, the mask slice of this worker into the row tile;
// barrier; B: scatter keep flags of the picks of each virtual worker; barrier; C: select mask or -inf in this worker's slice and write it back
static void top_k_mask_thread_picks(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_top_k_mask_context *             tctx  = (struct htp_top_k_mask_context *) data;
    const struct htp_top_k_mask_kernel_params * kp    = tctx->kparams;
    struct htp_ops_context *                    octx  = tctx->octx;
    dma_queue *                                 dma_q = th->dma_q;
    struct htp_thread_trace *                   tr    = th->trace;

    const struct htp_tensor * top_k = octx->src[1];
    const struct htp_tensor * score = octx->src[2];
    const struct htp_tensor * extra = octx->src[4];

    uint8_t *      slot       = tctx->rowin;
    const uint32_t tile_bytes = kp->tile_units * HTP_TOP_K_MASK_UNIT;
    uint8_t *      out        = tctx->tiles;
    uint8_t *      flags      = tctx->tiles + tile_bytes;

    const uint32_t n_virt = kp->n_hvx;

    // this worker's slice of the device columns: the mask goes straight into the row tile, it is popped after the scan.
    // The device share of the columns depends on mdev, so this slice is sized here.
    const uint32_t out_b0 = tctx->col0 * kp->elem_size;
    const uint32_t out_nb = tctx->ncols_out * kp->elem_size;
    const uint32_t wslice = hex_round_up(fastdiv(out_nb + nth - 1, &octx->n_hvx_div), 128);
    const uint32_t wb0    = out_b0 + ith * wslice;
    const uint32_t wnb    = ith * wslice < out_nb ? MIN(wslice, out_nb - ith * wslice) : 0;

    const uint32_t fslice = kp->u.picks.flags_slice_u * HTP_TOP_K_MASK_UNIT;

    for (uint32_t v = ith; v < n_virt; v += nth) {
        uint32_t n_pushed = 0;
        if (kp->blocks_bytes) {
            n_pushed += top_k_mask_push_slice(dma_q, v, (dma_addr_t) tctx->blocks, octx->src[3]->data, kp->blocks_bytes, kp->u.picks.blocks_slice_u * HTP_TOP_K_MASK_UNIT);
        }
        n_pushed += top_k_mask_push_slice(dma_q, v, (dma_addr_t) slot, top_k->data, kp->n_top * sizeof(int32_t), kp->u.picks.top_slice_u * HTP_TOP_K_MASK_UNIT);
        if (kp->flags & HTP_TOP_K_MASK_HAS_EXTRA) {
            n_pushed += top_k_mask_push_slice(dma_q, v, (dma_addr_t) (slot + tctx->off_extra), extra->data, kp->n_extra * sizeof(int32_t), kp->u.picks.extra_slice_u * HTP_TOP_K_MASK_UNIT);
        }
        if (kp->flags & HTP_TOP_K_MASK_HAS_SCORE) {
            n_pushed += top_k_mask_push_slice(dma_q, v, (dma_addr_t) (slot + tctx->off_score), score->data, kp->score_bytes, kp->u.picks.score_slice_u * HTP_TOP_K_MASK_UNIT);
        }

        // the mask slice follows the last preload in the queue, so the pops below leave it in flight
        if (wnb && v + nth >= n_virt) {
            dma_queue_push(dma_q, dma_make_data(out + wb0, octx->src[0]->data + wb0), wnb, wnb, wnb, 1);
        }

        const uint32_t fb0 = v * fslice;
        if (fb0 < tile_bytes) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) v);
            top_k_mask_fill(flags + fb0, MIN(fslice, tile_bytes - fb0), 0); // clear the flags
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) v);
        }

        for (uint32_t i = 0; i < n_pushed; i++) {
            top_k_mask_pop(dma_q, tr, (uint16_t) i);
        }
    }

    top_k_mask_barrier(&tctx->barrier, nth, tr, 0);

    HVX_Vector * tmp = (HVX_Vector *) (tctx->scratch + ith * HTP_TOP_K_MASK_SCRATCH_BYTES);

    for (uint32_t v = ith; v < n_virt; v += nth) {
        const uint32_t j0       = MIN(v * kp->u.picks.picks_per_hvx, kp->n_top);
        const uint32_t j1       = MIN(j0 + kp->u.picks.picks_per_hvx, kp->n_top);
        const bool     do_extra = v == n_virt - 1;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) v);
        if (kp->flags & HTP_TOP_K_MASK_F16) {
            top_k_mask_flag_tile_f16(tctx, slot, flags, tmp, j0, j1, do_extra);
        } else {
            top_k_mask_flag_tile_f32(tctx, slot, flags, tmp, j0, j1, do_extra);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) v);
    }

    top_k_mask_barrier(&tctx->barrier, 2 * nth, tr, 1);

    if (wnb) {
        top_k_mask_pop(dma_q, tr, (uint16_t) ith);
        // wb0 is 128B aligned (col0 is a multiple of 64 columns), the select may run past the row inside the tile
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
        top_k_mask_select(out + wb0, flags + wb0, hex_round_up(wnb, 128), kp->ninf);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
        dma_queue_push(dma_q, dma_make_data(octx->dst->data + wb0, out + wb0), wnb, wnb, wnb, 1);
        top_k_mask_pop(dma_q, tr, (uint16_t) ith);
    }
}

int op_top_k_mask(struct htp_ops_context * octx) {
    const struct htp_top_k_mask_kernel_params * kp = (const struct htp_top_k_mask_kernel_params *) octx->kernel_params;

    if (kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const bool picks = kp->kernel_type == HTP_TOP_K_MASK_KERNEL_PICKS;

    // mdev: ROWS gives this device a contiguous share of the rows, DMA writes are exempt from the 128B rule.
    // PICKS: every device scans the row, the write back is split by 64 column chunks.
    uint32_t dev_row0 = 0;
    uint32_t dev_rows = picks ? 1 : kp->u.rows.total_rows;
    uint32_t dev_c0   = 0;
    uint32_t dev_cols = kp->n_kv;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = picks ?
            htp_tensor_mdev_partition(kp->n_kv, 64, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div) :
            htp_tensor_mdev_partition(kp->u.rows.total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        if (range.count == 0) {
            return HTP_STATUS_OK;
        }
        if (picks) {
            dev_c0   = range.start;
            dev_cols = range.count;
        } else {
            dev_row0 = range.start;
            dev_rows = range.count;
        }
    }

    // fewer workers than kparams->n_hvx always fit the VTCM layout: ROWS runs one row range per worker, PICKS loops over the host slices
    const uint32_t max_hvx = MIN((uint32_t) kp->n_hvx, octx->ctx->n_hvx);
    if (!htp_ops_context_set_n_hvx(octx, picks ? max_hvx : MIN(max_hvx, dev_rows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t n_hvx = octx->n_hvx;

    uint8_t * vtcm = octx->ctx->vtcm_base;

    struct htp_top_k_mask_context tctx = {
        .octx      = octx,
        .kparams   = kp,
        .off_extra = kp->off_extra_units * HTP_TOP_K_MASK_UNIT,
        .off_score = kp->off_score_units * HTP_TOP_K_MASK_UNIT,
        .row0      = dev_row0,
        .row1      = dev_row0 + dev_rows,
        .col0      = dev_c0,
        .ncols_out = dev_cols,
        .blocks    = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_blocks_u * HTP_TOP_K_MASK_UNIT),
        .mask      = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_mask_u * HTP_TOP_K_MASK_UNIT),
        .rowin     = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_rowin_u * HTP_TOP_K_MASK_UNIT),
        .tiles     = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_tiles_u * HTP_TOP_K_MASK_UNIT),
        .mtiles    = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_mtiles_u * HTP_TOP_K_MASK_UNIT),
        .scratch   = VTCM_LAYOUT_PTR(uint8_t, vtcm, kp->off_scratch_u * HTP_TOP_K_MASK_UNIT),
    };

    if (picks) {
        atomic_init(&tctx.barrier, 0);
        work_queue_run(octx->ctx->hvx_queue, top_k_mask_thread_picks, &tctx, n_hvx);
        return HTP_STATUS_OK;
    }

    tctx.rows_per_hvx = fastdiv(dev_rows + n_hvx - 1, &octx->n_hvx_div);

    work_queue_run(octx->ctx->hvx_queue, top_k_mask_preload, &tctx, n_hvx);
    work_queue_run(octx->ctx->hvx_queue, top_k_mask_thread_rows, &tctx, n_hvx);

    return HTP_STATUS_OK;
}
