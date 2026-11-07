#ifndef HTP_TOP_K_MASK_OPS_H
#define HTP_TOP_K_MASK_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// HVX gather results land here (gathers write to VTCM only): 10 vectors per thread
#define HTP_TOP_K_MASK_SCRATCH_BYTES 1280

enum htp_top_k_mask_kernel {
    HTP_TOP_K_MASK_KERNEL_ROWS  = 0, // threads split rows, row inputs streamed per row
    HTP_TOP_K_MASK_KERNEL_PICKS = 1, // one row with one resident mask row: threads split the picks, keep flags select mask or -inf
};

#define HTP_TOP_K_MASK_F16           (1u << 0)
#define HTP_TOP_K_MASK_HAS_SCORE     (1u << 1)
#define HTP_TOP_K_MASK_HAS_BLOCKS    (1u << 2)
#define HTP_TOP_K_MASK_HAS_EXTRA     (1u << 3)
#define HTP_TOP_K_MASK_MASK_RESIDENT (1u << 4) // mask has exactly 1 row

#define HTP_TOP_K_MASK_UNIT 128 // unit of the compact u16 sizes in kernel_params

struct htp_top_k_mask_shape {
    uint32_t n_kv, elem_size, n_top, n_pick, n_blk, n_extra;
    uint32_t has_score, has_blocks, mask_resident;
};

struct htp_top_k_mask_vtcm_layout {
    size_t total_bytes;
    size_t off_blocks, off_mask, off_rowin, off_tiles, off_mtiles; // 0 when unused
    size_t off_scratch; // HTP_TOP_K_MASK_SCRATCH_BYTES per thread
    size_t rowin_bytes; // one slot of top_k + extra + score row, parts at htp_top_k_mask_rowin_off_*
    size_t tile_bytes;  // one output tile slot, 128B aligned
    size_t blocks_bytes, mask_bytes; // exact DMA sizes, 0 when unused
};

static inline uint64_t htp_top_k_mask_align128(uint64_t n) {
    return (n + 127) & ~(uint64_t) 127;
}

// Parts of one row input slot: top_k at 0, then extra, then score; used by the host layout
static inline uint64_t htp_top_k_mask_rowin_off_extra(uint32_t n_top) {
    return htp_top_k_mask_align128((uint64_t) n_top * 4);
}

static inline uint64_t htp_top_k_mask_rowin_off_score(uint32_t n_top, uint32_t n_extra) {
    return htp_top_k_mask_rowin_off_extra(n_top) + htp_top_k_mask_align128((uint64_t) n_extra * 4);
}

// the caller compares total_bytes against the VTCM size
static inline void htp_top_k_mask_vtcm_layout_build(struct htp_top_k_mask_vtcm_layout * L,
        const struct htp_top_k_mask_shape * S, uint32_t kernel_type, uint32_t n_threads, uint32_t tile_cols) {
    memset(L, 0, sizeof(*L));

    // PICKS streams the mask straight into its row tile
    const bool     picks        = kernel_type == HTP_TOP_K_MASK_KERNEL_PICKS;
    const uint64_t blocks_exact = S->has_blocks ? (uint64_t) S->n_blk * S->n_pick * 4 : 0;
    const uint64_t blocks_bytes = htp_top_k_mask_align128(blocks_exact);
    const uint64_t mask_exact   = S->mask_resident && !picks ? (uint64_t) S->n_kv * S->elem_size : 0;
    const uint64_t mask_bytes   = htp_top_k_mask_align128(mask_exact);
    const uint64_t rowin_bytes  = htp_top_k_mask_rowin_off_score(S->n_top, S->n_extra) +
                                  (S->has_score ? htp_top_k_mask_align128((uint64_t) S->n_pick * 4) : 0);
    const uint64_t tile_bytes   = htp_top_k_mask_align128((uint64_t) tile_cols * S->elem_size);

    // ROWS streams row inputs into 2 slots per thread. PICKS: one row, a row tile and a flag tile.
    const uint64_t rowin_total  = (picks ? 1 : 2 * (uint64_t) n_threads) * rowin_bytes;
    const uint64_t tiles_total  = picks ? 2 * tile_bytes : 2 * (uint64_t) n_threads * tile_bytes;
    const uint64_t mtiles_total = S->mask_resident ? 0 : tiles_total;
    const uint64_t scratch      = (uint64_t) n_threads * HTP_TOP_K_MASK_SCRATCH_BYTES;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_blocks, (size_t) blocks_bytes, blocks_bytes > 0);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_mask,   (size_t) mask_bytes,   mask_bytes > 0);
    VTCM_LAYOUT_ALLOC(off, off_rowin, (size_t) rowin_total);
    VTCM_LAYOUT_ALLOC(off, off_tiles, (size_t) tiles_total);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_mtiles, (size_t) mtiles_total, !S->mask_resident);
    VTCM_LAYOUT_ALLOC(off, off_scratch, (size_t) scratch);

    L->rowin_bytes  = (size_t) rowin_bytes;
    L->tile_bytes   = (size_t) tile_bytes;
    L->blocks_bytes = (size_t) blocks_exact;
    L->mask_bytes   = (size_t) mask_exact;
    L->total_bytes  = off;
}

// Set by the host, used as is by the NPU: no tensor shape, stride or layout is derived on the NPU.
// Sizes of one slot or tile, the slot parts and the VTCM offsets are in HTP_TOP_K_MASK_UNIT bytes as u16.
// Row strides: byte offset of flat row r of top_k, extra, score and dst is r * stride (the host requires row flattenable inputs).
// PICKS splits the work in n_threads virtual workers; fewer real workers loop over them.
struct htp_top_k_mask_kernel_params {
    uint8_t  n_threads, kernel_type, flags, elem_size;
    uint32_t vtcm_size;
    uint32_t n_kv, n_top, n_pick, n_extra;
    uint32_t ninf;                       // -inf cell pattern of elem_size
    uint32_t blocks_bytes, score_bytes;  // exact DMA sizes, gather regions are [0, bytes)
    uint32_t tile_cols, n_col_tiles;
    uint16_t n_blk, blocks_nb1;          // blocks_nb1 = n_blk * 4, fits a signed 16-bit multiply
    uint16_t off_blocks_u, off_mask_u, off_rowin_u, off_tiles_u, off_mtiles_u, off_scratch_u;
    uint16_t rowin_units, tile_units, off_extra_units, off_score_units;
    union {
        struct {
            uint32_t total_rows;
            uint32_t rs_top, rs_extra, rs_score, rs_dst; // flat row strides (bytes of row 1)
            uint32_t mask_nb1;                           // mask row stride, rows of the mask are dense above it
            union {
                uint32_t mask_bytes;                     // resident mask (1 row)
                struct {                                 // broadcast mask: row r -> (i1, i2, i3) -> mask row (i mod mne)
                    uint32_t mp[5];                      // fastdiv multipliers of dst ne1, ne2 and mask ne1, ne2, ne3
                    uint16_t ne[5];                      // the divisors
                    uint8_t  l[5];                       // fastdiv shifts
                } dec;
            } m;
        } rows;
        struct {
            uint32_t picks_per_thread;                   // per virtual worker, multiple of the lanes of one vector
            uint16_t blocks_slice_u, top_slice_u, extra_slice_u, score_slice_u, flags_slice_u; // per virtual worker slices
        } picks;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_top_k_mask_kernel_params) <= 128, "htp_top_k_mask_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_top_k_mask_kernel_params) <= 128, "htp_top_k_mask_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_TOP_K_MASK_OPS_H
