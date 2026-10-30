#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <stdatomic.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-types.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "top-k-mask-ops.h"

#define TOP_K_MASK_NINF_F32 0xff800000u
#define TOP_K_MASK_NINF_F16 0xfc00fc00u

struct htp_top_k_mask_context {
    struct htp_ops_context *                    octx;
    const struct htp_top_k_mask_kernel_params * kparams;

    uint32_t n_kv, n_top, n_pick, n_blk, n_extra, elem_size;
    uint32_t rows_per_thread;
    uint32_t row0, row1;                  // rows this device touches: whole rows for ROWS, rows covered by the item range for COLS
    uint32_t it0;                         // first item of this device
    uint32_t tile_cols, n_col_tiles;
    uint32_t items_per_thread, items_rem; // COLS split of this device's items across threads
    uint32_t row_bytes;                   // n_kv * elem_size
    uint32_t blocks_bytes;
    uint32_t mask_bytes;                  // resident mask row, 0 when the mask is per row
    uint32_t off_extra, off_score;        // parts inside one row input slot
    uint32_t blocks_nb1;
    uint32_t mask_resident;
    uint32_t ninf;
    uint32_t picks_per_thread;            // PICKS: picks per thread, a multiple of VLEN_FP32
    uint32_t col0, ncols_out;             // PICKS: columns this device writes back
    atomic_uint barrier;                  // PICKS: arrivals, barrier k waits for k * n_threads

    uint8_t * blocks;
    uint8_t * mask;
    uint8_t * rowin;
    uint8_t * tiles;
    uint8_t * mtiles;
    uint8_t * scratch;
};

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

static const int32_t top_k_mask_lane_idx[32] __attribute__((aligned(128))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
};

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
        HVX_Vector * restrict tmp, uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra, bool has_score, bool has_blocks,
        const bool is_f16, const bool flag) {
    const int32_t * restrict tk  = (const int32_t *) slot;
    const int32_t * restrict ex  = (const int32_t *) (slot + tctx->off_extra);
    const uint8_t * restrict sc  = slot + tctx->off_score;
    const uint8_t * restrict blk = tctx->blocks;

    const uint32_t n_blk   = tctx->n_blk;
    const uint32_t n_extra = do_extra ? tctx->n_extra : 0;
    const uint32_t mu      = ncols * tctx->elem_size - 1;
    const uint32_t sc_mu   = tctx->n_pick * sizeof(float) - 1;
    const uint32_t blk_mu  = tctx->blocks_bytes - 1;
    const uint32_t nb1_h   = (tctx->blocks_nb1 << 16) | tctx->blocks_nb1;

    const HVX_Vector v_c0    = Q6_V_vsplat_R(c0);
    const HVX_Vector v_ncols = Q6_V_vsplat_R(ncols);
    const HVX_Vector v_npick = Q6_V_vsplat_R(tctx->n_pick);
    const HVX_Vector v_pmax  = Q6_V_vsplat_R(tctx->n_pick - 1); // dead picks gather the last entry, every gather stays in its region
    const HVX_Vector v_lane  = *(const HVX_Vector *) top_k_mask_lane_idx;
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
        // 4 pick vectors per batch, the vectors past j1 are dead
        for (uint32_t j = j0; j < j1; j += 4 * VLEN_FP32) {
            HVX_Vector c0v = top_k_mask_load_picks(tk, j + 0 * VLEN_FP32, v_rem, v_npick);
            HVX_Vector c1v = top_k_mask_load_picks(tk, j + 1 * VLEN_FP32, v_rem, v_npick);
            HVX_Vector c2v = top_k_mask_load_picks(tk, j + 2 * VLEN_FP32, v_rem, v_npick);
            HVX_Vector c3v = top_k_mask_load_picks(tk, j + 3 * VLEN_FP32, v_rem, v_npick);
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
                                        uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra, bool has_score, bool has_blocks) {
    top_k_mask_scatter_tile(tctx, slot, mrow, tile, tmp, c0, ncols, j0, j1, do_extra, has_score, has_blocks, false, false);
}

static __attribute__((noinline)) void top_k_mask_scatter_tile_f16(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        const uint8_t * restrict mrow, uint8_t * restrict tile, HVX_Vector * restrict tmp,
                                        uint32_t c0, uint32_t ncols, uint32_t j0, uint32_t j1, bool do_extra, bool has_score, bool has_blocks) {
    top_k_mask_scatter_tile(tctx, slot, mrow, tile, tmp, c0, ncols, j0, j1, do_extra, has_score, has_blocks, true, false);
}

// PICKS: keep flags (all ones) of the picks [j0, j1) into the flag tile of the whole row, the element size picks the flag width
static __attribute__((noinline)) void top_k_mask_flag_tile_f32(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        uint8_t * restrict flags, HVX_Vector * restrict tmp, uint32_t j0, uint32_t j1, bool do_extra,
                                        bool has_score, bool has_blocks) {
    top_k_mask_scatter_tile(tctx, slot, NULL, flags, tmp, 0, tctx->n_kv, j0, j1, do_extra, has_score, has_blocks, false, true);
}

static __attribute__((noinline)) void top_k_mask_flag_tile_f16(const struct htp_top_k_mask_context * tctx, const uint8_t * restrict slot,
                                        uint8_t * restrict flags, HVX_Vector * restrict tmp, uint32_t j0, uint32_t j1, bool do_extra,
                                        bool has_score, bool has_blocks) {
    top_k_mask_scatter_tile(tctx, slot, NULL, flags, tmp, 0, tctx->n_kv, j0, j1, do_extra, has_score, has_blocks, true, true);
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

static inline void top_k_mask_row_index(const struct htp_top_k_mask_context * tctx, uint32_t r,
                                        uint32_t * i1, uint32_t * i2, uint32_t * i3) {
    const struct htp_top_k_mask_kernel_params * kp  = tctx->kparams;
    const struct htp_tensor *                   dst = tctx->octx->dst;

    const uint32_t ne1  = dst->ne[1];
    const uint32_t ne12 = dst->ne[1] * dst->ne[2];

    *i3 = fastdiv(r, &kp->div_ne12);
    const uint32_t rem = r - *i3 * ne12;
    *i2 = fastdiv(rem, &kp->div_ne1);
    *i1 = rem - *i2 * ne1;
}

// Push the row inputs of row (i1, i2, i3) into slot: top_k, extra, score. Returns the push count.
static inline uint32_t top_k_mask_push_rowin(const struct htp_top_k_mask_context * tctx, dma_queue * dma_q,
                                             uint32_t i1, uint32_t i2, uint32_t i3, uint8_t * slot) {
    const struct htp_ops_context * octx  = tctx->octx;
    const struct htp_tensor *      top_k = octx->src[1];
    const struct htp_tensor *      score = octx->src[2];
    const struct htp_tensor *      extra = octx->src[4];

    const uint32_t tk_bytes = tctx->n_top * sizeof(int32_t);
    const uint32_t tk_off   = i1 * top_k->nb[1] + i2 * top_k->nb[2] + i3 * top_k->nb[3];
    dma_queue_push(dma_q, dma_make_data(slot, top_k->data + tk_off), tk_bytes, tk_bytes, tk_bytes, 1);
    uint32_t n = 1;

    if (extra) {
        const uint32_t ex_bytes = tctx->n_extra * sizeof(int32_t);
        const uint32_t ex_off   = i1 * extra->nb[1] + i2 * extra->nb[2] + i3 * extra->nb[3];
        dma_queue_push(dma_q, dma_make_data(slot + tctx->off_extra, extra->data + ex_off), ex_bytes, ex_bytes, ex_bytes, 1);
        n++;
    }

    if (score) {
        const uint32_t sc_bytes = tctx->n_pick * sizeof(float);
        const uint32_t sc_off   = i1 * score->nb[1] + i2 * score->nb[2] + i3 * score->nb[3];
        dma_queue_push(dma_q, dma_make_data(slot + tctx->off_score, score->data + sc_off), sc_bytes, sc_bytes, sc_bytes, 1);
        n++;
    }

    return n;
}

// Push the inputs of item it (row r, col tile from c0): ROWS row inputs on the first tile of a row, then the mask tile. Returns the push count.
static inline uint32_t top_k_mask_push_item(const struct htp_top_k_mask_context * tctx, dma_queue * dma_q, uint32_t it,
                                            uint8_t * slot, uint8_t * mslot, bool cols) {
    const struct htp_ops_context *              octx = tctx->octx;
    const struct htp_top_k_mask_kernel_params * kp   = tctx->kparams;
    const struct htp_tensor *                   mask = octx->src[0];

    const uint32_t r  = fastdiv(it, &kp->div_n_col_tiles);
    const uint32_t c0 = (it - r * tctx->n_col_tiles) * tctx->tile_cols;

    uint32_t i1, i2, i3;
    top_k_mask_row_index(tctx, r, &i1, &i2, &i3);

    uint32_t n = 0;
    if (!cols && c0 == 0) {
        n += top_k_mask_push_rowin(tctx, dma_q, i1, i2, i3, slot);
    }

    if (mslot) {
        const uint32_t m1    = fastmodulo(i1, mask->ne[1], &kp->div_m1);
        const uint32_t m2    = fastmodulo(i2, mask->ne[2], &kp->div_m2);
        const uint32_t m3    = fastmodulo(i3, mask->ne[3], &kp->div_m3);
        const uint32_t m_off = m1 * mask->nb[1] + m2 * mask->nb[2] + m3 * mask->nb[3] + c0 * tctx->elem_size;
        const uint32_t nb    = MIN(tctx->tile_cols, tctx->n_kv - c0) * tctx->elem_size;
        dma_queue_push(dma_q, dma_make_data(mslot, mask->data + m_off), nb, nb, nb, 1);
        n++;
    }

    return n;
}

// Push this thread's 128B aligned slice of a contiguous n byte copy. Returns the push count.
static inline uint32_t top_k_mask_push_slice(const struct htp_ops_context * octx, dma_queue * dma_q, unsigned int nth, unsigned int ith,
                                             dma_addr_t dst, dma_addr_t src, uint32_t n) {
    const uint32_t slice = hex_round_up(fastdiv(n + nth - 1, &octx->n_threads_div), 128);
    const uint32_t b0    = ith * slice;
    if (b0 >= n) {
        return 0;
    }
    const uint32_t nb = MIN(slice, n - b0);
    dma_queue_push(dma_q, dma_make_data(dst + b0, src + b0), nb, nb, nb, 1);
    return 1;
}

// Pass 1: load the blocks table, the resident mask row and (COLS) the row inputs of every row, split across threads.
// Fill the tiles of this thread with -inf meanwhile.
static void top_k_mask_preload(unsigned int nth, unsigned int ith, void * data) {
    struct htp_top_k_mask_context * tctx  = (struct htp_top_k_mask_context *) data;
    struct htp_ops_context *        octx  = tctx->octx;
    dma_queue *                     dma_q = octx->ctx->dma[ith];

    uint32_t n_pushed = 0;

    if (tctx->blocks_bytes) {
        n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) tctx->blocks, octx->src[3]->data, tctx->blocks_bytes);
    }
    if (tctx->mask_bytes) {
        n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) tctx->mask, octx->src[0]->data, tctx->mask_bytes);
    }

    if (tctx->kparams->kernel_type == HTP_TOP_K_MASK_KERNEL_COLS) {
        const struct htp_tensor * top_k = octx->src[1];
        const struct htp_tensor * score = octx->src[2];
        const struct htp_tensor * extra = octx->src[4];

        for (uint32_t r = tctx->row0; r < tctx->row1; r++) {
            uint8_t * slot = tctx->rowin + r * tctx->kparams->rowin_bytes;

            uint32_t i1, i2, i3;
            top_k_mask_row_index(tctx, r, &i1, &i2, &i3);

            const uint32_t tk_off = i1 * top_k->nb[1] + i2 * top_k->nb[2] + i3 * top_k->nb[3];
            n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) slot, top_k->data + tk_off, tctx->n_top * sizeof(int32_t));

            if (extra) {
                const uint32_t ex_off = i1 * extra->nb[1] + i2 * extra->nb[2] + i3 * extra->nb[3];
                n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) (slot + tctx->off_extra), extra->data + ex_off, tctx->n_extra * sizeof(int32_t));
            }
            if (score) {
                const uint32_t sc_off = i1 * score->nb[1] + i2 * score->nb[2] + i3 * score->nb[3];
                n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) (slot + tctx->off_score), score->data + sc_off, tctx->n_pick * sizeof(float));
            }
        }
    }

    const uint32_t tile_bytes = tctx->kparams->tile_bytes;
    top_k_mask_fill(tctx->tiles + ith * 2 * tile_bytes, 2 * tile_bytes, tctx->ninf);

    for (uint32_t i = 0; i < n_pushed; i++) {
        dma_queue_pop(dma_q);
    }
}

// Depth-2 DMA pipeline over items [it0, it1), item = row * n_col_tiles + col tile. ROWS ranges hold whole rows.
static inline void top_k_mask_run_items(const struct htp_top_k_mask_context * tctx, unsigned int ith, uint32_t it0, uint32_t it1, bool cols) {
    struct htp_ops_context *                    octx  = tctx->octx;
    const struct htp_top_k_mask_kernel_params * kp    = tctx->kparams;
    const struct htp_tensor *                   dst   = octx->dst;
    dma_queue *                                 dma_q = octx->ctx->dma[ith];

    const bool     has_score  = octx->src[2] != NULL;
    const bool     has_blocks = octx->src[3] != NULL;
    const bool     is_f16     = tctx->elem_size == sizeof(__fp16);
    const uint32_t elem_size  = tctx->elem_size;

    const uint32_t rowin_bytes = kp->rowin_bytes;
    const uint32_t tile_bytes  = kp->tile_bytes;
    const uint32_t tile_cols   = tctx->tile_cols;
    const uint32_t n_col_tiles = tctx->n_col_tiles;

    // ROWS: 2 row input slots per thread, alternating by row. COLS: one preloaded slot per row.
    uint8_t * rowin  = cols ? tctx->rowin : tctx->rowin + ith * 2 * rowin_bytes;
    uint8_t * tiles  = tctx->tiles + ith * 2 * tile_bytes;
    uint8_t * mtiles = tctx->mask_resident ? NULL : tctx->mtiles + ith * 2 * tile_bytes;

    HVX_Vector * tmp = (HVX_Vector *) (tctx->scratch + ith * HTP_TOP_K_MASK_SCRATCH_BYTES);

    const dma_addr_t dst_data = dst->data;
    const uint32_t   row0     = fastdiv(it0, &kp->div_n_col_tiles);

    uint32_t n_in[2] = { 0, 0 };

    for (uint32_t it = it0, s = 0; it < it1 && s < 2; it++, s++) {
        // Dummy dst writeback to establish queue ordering
        dma_queue_push(dma_q, dma_make_data(dst_data, tiles + s * tile_bytes), 0, 0, 0, 0);
        const uint32_t r = fastdiv(it, &kp->div_n_col_tiles);
        n_in[s] = top_k_mask_push_item(tctx, dma_q, it, rowin + ((r - row0) & 1) * rowin_bytes, mtiles ? mtiles + s * tile_bytes : NULL, cols);
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t it = it0; it < it1; it++) {
        const uint32_t s    = (it - it0) & 1;
        uint8_t *      tile = tiles + s * tile_bytes;

        // writeback of item it - 2 (or the dummy), then the tile is free again
        dma_queue_pop(dma_q);
        if (it - it0 >= 2) {
            top_k_mask_fill(tile, tile_bytes, tctx->ninf);
        }

        for (uint32_t i = 0; i < n_in[s]; i++) {
            dma_queue_pop(dma_q);
        }

        const uint32_t r     = fastdiv(it, &kp->div_n_col_tiles);
        const uint32_t c0    = (it - r * n_col_tiles) * tile_cols;
        const uint32_t ncols = MIN(tile_cols, tctx->n_kv - c0);

        const uint8_t * slot = cols ? rowin + r * rowin_bytes : rowin + ((r - row0) & 1) * rowin_bytes;
        const uint8_t * mrow = mtiles ? mtiles + s * tile_bytes : tctx->mask + c0 * elem_size;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);
        if (is_f16) {
            top_k_mask_scatter_tile_f16(tctx, slot, mrow, tile, tmp, c0, ncols, 0, tctx->n_top, true, has_score, has_blocks);
        } else {
            top_k_mask_scatter_tile_f32(tctx, slot, mrow, tile, tmp, c0, ncols, 0, tctx->n_top, true, has_score, has_blocks);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) it);

        uint32_t i1, i2, i3;
        top_k_mask_row_index(tctx, r, &i1, &i2, &i3);
        const uint32_t d_off = i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3] + c0 * elem_size;
        const uint32_t nb    = ncols * elem_size;
        dma_queue_push(dma_q, dma_make_data(dst_data + d_off, tile), nb, nb, nb, 1);

        if (it + 2 < it1) {
            const uint32_t rn = fastdiv(it + 2, &kp->div_n_col_tiles);
            n_in[s] = top_k_mask_push_item(tctx, dma_q, it + 2, rowin + ((rn - row0) & 1) * rowin_bytes, mtiles ? mtiles + s * tile_bytes : NULL, cols);
        }
    }

    const uint32_t n_wb = MIN(it1 - it0, 2);
    for (uint32_t i = 0; i < n_wb; i++) {
        dma_queue_pop(dma_q);
    }
}

// Pass 2 (ROWS): whole rows split across threads, the col tiles of a row in order
static void top_k_mask_thread_rows(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_top_k_mask_context * tctx = (const struct htp_top_k_mask_context *) data;

    const uint32_t ir0 = tctx->row0 + ith * tctx->rows_per_thread;
    const uint32_t ir1 = MIN(ir0 + tctx->rows_per_thread, tctx->row1);
    if (ir0 >= ir1) {
        return;
    }

    top_k_mask_run_items(tctx, ith, ir0 * tctx->n_col_tiles, ir1 * tctx->n_col_tiles, false);
}

// Pass 2 (COLS): (row, col tile) items split contiguously across threads, the first items_rem threads take one extra item
static void top_k_mask_thread_cols(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_top_k_mask_context * tctx = (const struct htp_top_k_mask_context *) data;

    const uint32_t it0 = tctx->it0 + ith * tctx->items_per_thread + MIN(ith, tctx->items_rem);
    const uint32_t it1 = it0 + tctx->items_per_thread + (ith < tctx->items_rem);
    if (it0 >= it1) {
        return;
    }

    top_k_mask_run_items(tctx, ith, it0, it1, true);
}

static inline void top_k_mask_barrier(atomic_uint * b, uint32_t target) {
    atomic_fetch_add_explicit(b, 1, memory_order_release);
    while (atomic_load_explicit(b, memory_order_acquire) < target) {
    }
}

// PICKS, one pass: shared preload, the mask slice of this thread into the row tile, zero a slice of the flag tile;
// barrier; scatter keep flags of this thread's picks; barrier; select mask or -inf in this thread's slice and write it back
static void top_k_mask_thread_picks(unsigned int nth, unsigned int ith, void * data) {
    struct htp_top_k_mask_context * tctx  = (struct htp_top_k_mask_context *) data;
    struct htp_ops_context *        octx  = tctx->octx;
    dma_queue *                     dma_q = octx->ctx->dma[ith];
    struct htp_thread_trace *       tr    = &octx->ctx->trace[ith];

    const struct htp_tensor * top_k = octx->src[1];
    const struct htp_tensor * score = octx->src[2];
    const struct htp_tensor * extra = octx->src[4];

    uint8_t *      slot       = tctx->rowin;
    const uint32_t tile_bytes = tctx->kparams->tile_bytes;
    uint8_t *      out        = tctx->tiles;
    uint8_t *      flags      = tctx->tiles + tile_bytes;

    uint32_t n_pushed = 0;
    if (tctx->blocks_bytes) {
        n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) tctx->blocks, octx->src[3]->data, tctx->blocks_bytes);
    }
    n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) slot, top_k->data, tctx->n_top * sizeof(int32_t));
    if (extra) {
        n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) (slot + tctx->off_extra), extra->data, tctx->n_extra * sizeof(int32_t));
    }
    if (score) {
        n_pushed += top_k_mask_push_slice(octx, dma_q, nth, ith, (dma_addr_t) (slot + tctx->off_score), score->data, tctx->n_pick * sizeof(float));
    }

    // this thread's slice of the device columns: the mask goes straight into the row tile, it is popped after the scan
    const uint32_t out_b0 = tctx->col0 * tctx->elem_size;
    const uint32_t out_nb = tctx->ncols_out * tctx->elem_size;
    const uint32_t wslice = hex_round_up(fastdiv(out_nb + nth - 1, &octx->n_threads_div), 128);
    const uint32_t wb0    = out_b0 + ith * wslice;
    const uint32_t wnb    = ith * wslice < out_nb ? MIN(wslice, out_nb - ith * wslice) : 0;
    if (wnb) {
        dma_queue_push(dma_q, dma_make_data(out + wb0, octx->src[0]->data + wb0), wnb, wnb, wnb, 1);
    }

    const uint32_t fslice = hex_round_up(fastdiv(tile_bytes + nth - 1, &octx->n_threads_div), 128);
    const uint32_t fb0    = ith * fslice;
    if (fb0 < tile_bytes) {
        top_k_mask_fill(flags + fb0, MIN(fslice, tile_bytes - fb0), 0); // clear the flags
    }

    for (uint32_t i = 0; i < n_pushed; i++) {
        dma_queue_pop(dma_q);
    }

    top_k_mask_barrier(&tctx->barrier, nth);

    const uint32_t j0       = MIN(ith * tctx->picks_per_thread, tctx->n_top);
    const uint32_t j1       = MIN(j0 + tctx->picks_per_thread, tctx->n_top);
    const bool     do_extra = ith == nth - 1;

    HVX_Vector * tmp = (HVX_Vector *) (tctx->scratch + ith * HTP_TOP_K_MASK_SCRATCH_BYTES);

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
    if (tctx->elem_size == sizeof(__fp16)) {
        top_k_mask_flag_tile_f16(tctx, slot, flags, tmp, j0, j1, do_extra, score != NULL, octx->src[3] != NULL);
    } else {
        top_k_mask_flag_tile_f32(tctx, slot, flags, tmp, j0, j1, do_extra, score != NULL, octx->src[3] != NULL);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    top_k_mask_barrier(&tctx->barrier, 2 * nth);

    if (wnb) {
        dma_queue_pop(dma_q);
        // wb0 is 128B aligned (col0 is a multiple of 64 columns), the select may run past the row inside the tile
        top_k_mask_select(out + wb0, flags + wb0, hex_round_up(wnb, 128), tctx->ninf);
        dma_queue_push(dma_q, dma_make_data(octx->dst->data + wb0, out + wb0), wnb, wnb, wnb, 1);
        dma_queue_pop(dma_q);
    }
}

int op_top_k_mask(struct htp_ops_context * octx) {
    const struct htp_tensor * mask   = octx->src[0];
    const struct htp_tensor * top_k  = octx->src[1];
    const struct htp_tensor * score  = octx->src[2];
    const struct htp_tensor * blocks = octx->src[3];
    const struct htp_tensor * extra  = octx->src[4];
    const struct htp_tensor * dst    = octx->dst;

    const struct htp_top_k_mask_kernel_params * kp = (const struct htp_top_k_mask_kernel_params *) octx->kernel_params;

    const bool cols  = kp->kernel_type == HTP_TOP_K_MASK_KERNEL_COLS;
    const bool picks = kp->kernel_type == HTP_TOP_K_MASK_KERNEL_PICKS;
    if ((dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32) || (!cols && !picks && kp->kernel_type != HTP_TOP_K_MASK_KERNEL_ROWS)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t total_rows   = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const uint32_t n_items_full = total_rows * kp->n_col_tiles;

    // mdev: this device takes a contiguous share of the rows (ROWS) or of the (row, col tile) items (COLS), DMA writes are exempt from the 128B rule
    uint32_t dev_it0   = 0;
    uint32_t dev_items = n_items_full;
    if (octx->ctx->mdev.count > 1 && !picks) {
        const struct htp_tensor_mdev_range range = cols ?
            htp_tensor_mdev_partition(n_items_full, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div) :
            htp_tensor_mdev_partition(total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        dev_it0   = cols ? range.start : range.start * kp->n_col_tiles;
        dev_items = cols ? range.count : range.count * kp->n_col_tiles;
    }

    if (dev_items == 0) {
        return HTP_STATUS_OK;
    }

    // PICKS: every device scans the row, the write back is split by 64 column chunks
    uint32_t dev_c0    = 0;
    uint32_t dev_ncols = dst->ne[0];
    if (picks && octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(dst->ne[0], 64, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        if (range.count == 0) {
            return HTP_STATUS_OK;
        }
        dev_c0    = range.start;
        dev_ncols = range.count;
    }

    // fewer threads than kparams->n_threads always fit the VTCM layout
    const uint32_t dev_rows = cols ? 0 : dev_items / kp->n_col_tiles;
    if (!htp_ops_context_set_n_threads(octx, picks ? kp->n_threads : MIN(kp->n_threads, cols ? dev_items : dev_rows))) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t n_threads = octx->n_threads;

    const uint32_t elem_size = dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t n_kv      = dst->ne[0];
    const uint32_t n_top     = top_k->ne[0];
    const uint32_t n_extra   = extra ? extra->ne[0] : 0;

    struct htp_top_k_mask_context tctx = {
        .octx          = octx,
        .kparams       = kp,
        .n_kv          = n_kv,
        .n_top         = n_top,
        .n_pick        = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv,
        .n_blk         = blocks ? blocks->ne[0] : 0,
        .n_extra       = n_extra,
        .elem_size     = elem_size,
        .it0           = dev_it0,
        .tile_cols     = kp->tile_cols,
        .n_col_tiles   = kp->n_col_tiles,
        .row_bytes     = n_kv * elem_size,
        .blocks_bytes  = blocks ? blocks->ne[1] * blocks->nb[1] : 0,
        .mask_resident = mask->ne[1] * mask->ne[2] * mask->ne[3] == 1,
        .off_extra     = (uint32_t) htp_top_k_mask_rowin_off_extra(n_top),
        .off_score     = (uint32_t) htp_top_k_mask_rowin_off_score(n_top, n_extra),
        .blocks_nb1    = blocks ? blocks->nb[1] : 0,
        .ninf          = elem_size == sizeof(__fp16) ? TOP_K_MASK_NINF_F16 : TOP_K_MASK_NINF_F32,
        .blocks        = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_blocks),
        .mask          = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_mask),
        .rowin         = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_rowin),
        .tiles         = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_tiles),
        .mtiles        = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_mtiles),
        .scratch       = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, kp->off_scratch),
    };

    tctx.mask_bytes      = tctx.mask_resident ? tctx.row_bytes : 0;
    // ROWS: whole rows. COLS: the rows covered by the item range, only those get preloaded
    tctx.row0 = fastdiv(dev_it0, &kp->div_n_col_tiles);
    tctx.row1 = fastdiv(dev_it0 + dev_items - 1, &kp->div_n_col_tiles) + 1;

    tctx.rows_per_thread = fastdiv(dev_rows + n_threads - 1, &octx->n_threads_div);
    if (cols) {
        tctx.items_per_thread = fastdiv(dev_items, &octx->n_threads_div);
        tctx.items_rem        = dev_items - tctx.items_per_thread * n_threads;
    }

    if (picks) {
        tctx.picks_per_thread = hex_round_up(fastdiv(n_top + n_threads - 1, &octx->n_threads_div), VLEN_FP32);
        tctx.col0             = dev_c0;
        tctx.ncols_out        = dev_ncols;
        atomic_init(&tctx.barrier, 0);
        work_queue_run(octx->ctx->work_queue, top_k_mask_thread_picks, &tctx, n_threads);
        return HTP_STATUS_OK;
    }

    work_queue_run(octx->ctx->work_queue, top_k_mask_preload, &tctx, n_threads);
    work_queue_run(octx->ctx->work_queue, cols ? top_k_mask_thread_cols : top_k_mask_thread_rows, &tctx, n_threads);

    return HTP_STATUS_OK;
}
